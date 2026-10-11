"""Causal estimators for experiments with multiple treatment arms.

The module provides S-Learner and T-Learner random forests, HydraNet, and
MEDUSA. Each estimator implements ``fit(X, T, Y)`` and
``predict_potential_outcomes(X)``; predictions have shape ``(n_samples, K)``.

The controlled validation runner imports these estimators from this sibling
module. Random-forest estimators require NumPy, scikit-learn, and joblib;
HydraNet and MEDUSA also require PyTorch.

The MEDUSA network architecture itself is defined once in ``medusa.py`` and
imported here; this module keeps the exact MEDUSA training configuration used
by the validation suite. ``medusa.py`` additionally ships a full-featured
estimator with an optional doubly robust moment penalty and Hajek-AIPW
population estimates.
"""

from __future__ import annotations

import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor


_SKLEARN_PARALLEL_FALSE_POSITIVE = (
    "`sklearn.utils.parallel.delayed` should be used with "
    "`sklearn.utils.parallel.Parallel` to make it possible to propagate "
    "the scikit-learn configuration of the current thread to the joblib workers."
)
_RF_WARNING_HOOK_LOCK = threading.Lock()
_RF_WARNING_HOOK_DEPTH = 0
_RF_ORIGINAL_SHOWWARNING = None


def _showwarning_without_sklearn_parallel_false_positive(
    message, category, filename, lineno, file=None, line=None
):
    """Suppress one known scikit-learn warning and forward all others."""
    if (
        issubclass(category, UserWarning)
        and str(message) == _SKLEARN_PARALLEL_FALSE_POSITIVE
    ):
        return
    if _RF_ORIGINAL_SHOWWARNING is not None:
        _RF_ORIGINAL_SHOWWARNING(
            message, category, filename, lineno, file=file, line=line
        )


@contextmanager
def _random_forest_parallel_context():
    """Keep random forests parallel while filtering a known warning race.

    Python 3.14 worker contexts can lose ordinary warning filters. This
    exact-message hook also covers worker threads. Random forests use joblib's
    threading backend, and all unrelated warnings pass through unchanged.
    """
    global _RF_WARNING_HOOK_DEPTH, _RF_ORIGINAL_SHOWWARNING
    with _RF_WARNING_HOOK_LOCK:
        if _RF_WARNING_HOOK_DEPTH == 0:
            _RF_ORIGINAL_SHOWWARNING = warnings.showwarning
            warnings.showwarning = _showwarning_without_sklearn_parallel_false_positive
        _RF_WARNING_HOOK_DEPTH += 1
    try:
        with parallel_backend("threading"):
            yield
    finally:
        with _RF_WARNING_HOOK_LOCK:
            _RF_WARNING_HOOK_DEPTH -= 1
            if _RF_WARNING_HOOK_DEPTH == 0:
                if _RF_ORIGINAL_SHOWWARNING is not None:
                    warnings.showwarning = _RF_ORIGINAL_SHOWWARNING
                _RF_ORIGINAL_SHOWWARNING = None


class BaseCausalModel:
    """Common interface for multi-treatment causal estimators."""

    name: str
    K: int

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """Fit the estimator using features, treatment indices, and outcomes."""
        raise NotImplementedError

    def predict_potential_outcomes(self, X: np.ndarray) -> np.ndarray:
        """Return an ``(n_samples, K)`` matrix of potential-outcome estimates."""
        raise NotImplementedError


class SLearnerMulti(BaseCausalModel):
    """Estimate all treatment outcomes with one shared prediction model.

    The estimator fits ``[X, one_hot(T)] -> Y`` once, then predicts each
    potential outcome by replacing the observed treatment with each arm.
    """

    def __init__(self, base_model=None, K: int = 4, name: str = "S-Learner RF"):
        self.K = K
        self.name = name
        self.base_model = base_model if base_model is not None else RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
        self.model = clone(self.base_model)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        X = np.asarray(X)
        T = np.asarray(T, dtype=int)
        Y = np.asarray(Y)

        T_onehot = np.eye(self.K)[T]
        XT = np.hstack([X, T_onehot])
        with _random_forest_parallel_context():
            self.model.fit(XT, Y)
        return self

    def predict_potential_outcomes(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        preds = []
        with _random_forest_parallel_context():
            for t in range(self.K):
                Tt = np.zeros((len(X), self.K))
                Tt[:, t] = 1.0
                XT = np.hstack([X, Tt])
                preds.append(self.model.predict(XT))
        return np.column_stack(preds)


class TLearnerMulti(BaseCausalModel):
    """Estimate each treatment outcome with a separate prediction model.

    Each model is fitted only on observations assigned to its treatment arm.
    """

    def __init__(self, base_model=None, K: int = 4, name: str = "T-Learner RF"):
        self.K = K
        self.name = name
        self.base_model = base_model if base_model is not None else RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
        self.models = [clone(self.base_model) for _ in range(self.K)]

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        X = np.asarray(X)
        T = np.asarray(T, dtype=int)
        Y = np.asarray(Y)

        with _random_forest_parallel_context():
            for t in range(self.K):
                idx = T == t
                if idx.sum() == 0:
                    raise ValueError(f"Treatment {t} is missing from the training data.")
                self.models[t].fit(X[idx], Y[idx])
        return self

    def predict_potential_outcomes(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        with _random_forest_parallel_context():
            predictions = [model.predict(X) for model in self.models]
        return np.column_stack(predictions)


# HydraNet model.

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None

if nn is not None:
    try:
        from medusa import MEDUSA
    except ImportError:
        # This file is sometimes loaded by path (importlib) from another
        # working directory; fall back to loading the sibling medusa.py.
        import importlib.util as _importlib_util
        import sys as _sys
        from pathlib import Path as _Path

        _medusa_path = _Path(__file__).resolve().with_name("medusa.py")
        _medusa_spec = _importlib_util.spec_from_file_location(
            "controlled_validation_medusa", _medusa_path
        )
        _medusa_module = _importlib_util.module_from_spec(_medusa_spec)
        _sys.modules[_medusa_spec.name] = _medusa_module
        _medusa_spec.loader.exec_module(_medusa_module)
        MEDUSA = _medusa_module.MEDUSA
else:
    MEDUSA = None


if nn is not None:
    class _HydraNetTorch(nn.Module):
        def __init__(self, p: int, K: int, hidden: int = 128, depth: int = 2, dropout: float = 0.0):
            super().__init__()
            self.K = K

            layers = []
            in_dim = p
            for _ in range(depth):
                layers.append(nn.Linear(in_dim, hidden))
                layers.append(nn.ELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = hidden
            self.rep = nn.Sequential(*layers)

            self.outcome_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ELU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(K)
            ])

            self.propensity = nn.Linear(hidden, K)

        def forward(self, x):
            z = self.rep(x)
            mu = torch.cat([head(z) for head in self.outcome_heads], dim=1)
            logits = self.propensity(z)
            return mu, logits


@dataclass
class HydraNetConfig:
    """Hyperparameters for the HydraNet architecture and training loop."""

    hidden: int = 128
    depth: int = 2
    dropout: float = 0.0
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    alpha_propensity: float = 1.0
    verbose: bool = False
    device: Optional[str] = None
    random_state: int = 42


class HydraNetCausalModel(BaseCausalModel):
    """Multi-treatment HydraNet estimator implemented with PyTorch.

    The training objective combines factual-outcome MSE with treatment
    assignment cross-entropy, weighted by ``alpha_propensity``.

    ``predict_potential_outcomes(X)`` returns all ``K`` treatment heads.
    """

    def __init__(self, K: int, config: Optional[HydraNetConfig] = None, name: str = "HydraNet"):
        if torch is None:
            raise ImportError("PyTorch is not installed. Install it with: pip install torch")
        self.K = K
        self.name = name
        self.config = config or HydraNetConfig()
        self.model = None
        self.x_mean_ = None
        self.x_std_ = None
        self.y_mean_ = None
        self.y_std_ = None

    def _device(self):
        if self.config.device is not None:
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _standardize_X_fit(self, X):
        self.x_mean_ = X.mean(axis=0, keepdims=True)
        self.x_std_ = X.std(axis=0, keepdims=True) + 1e-8
        return (X - self.x_mean_) / self.x_std_

    def _standardize_X_transform(self, X):
        return (X - self.x_mean_) / self.x_std_

    def _standardize_Y_fit(self, Y):
        self.y_mean_ = float(Y.mean())
        self.y_std_ = float(Y.std() + 1e-8)
        return (Y - self.y_mean_) / self.y_std_

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        cfg = self.config
        np.random.seed(cfg.random_state)
        torch.manual_seed(cfg.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_state)

        X = np.asarray(X, dtype=np.float32)
        T = np.asarray(T, dtype=np.int64)
        Y = np.asarray(Y, dtype=np.float32).reshape(-1)

        Xs = self._standardize_X_fit(X).astype(np.float32)
        Ys = self._standardize_Y_fit(Y).astype(np.float32)

        device = self._device()
        self.model = _HydraNetTorch(
            p=X.shape[1],
            K=self.K,
            hidden=cfg.hidden,
            depth=cfg.depth,
            dropout=cfg.dropout,
        ).to(device)

        dataset = TensorDataset(
            torch.tensor(Xs, dtype=torch.float32),
            torch.tensor(T, dtype=torch.long),
            torch.tensor(Ys, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        last_log_time = time.perf_counter()
        for epoch in range(cfg.epochs):
            self.model.train()
            total_loss = 0.0
            for xb, tb, yb in loader:
                xb = xb.to(device)
                tb = tb.to(device)
                yb = yb.to(device)

                mu, logits = self.model(xb)
                mu_obs = mu.gather(1, tb.view(-1, 1)).squeeze(1)

                loss_y = F.mse_loss(mu_obs, yb)
                loss_t = F.cross_entropy(logits, tb)
                loss = loss_y + cfg.alpha_propensity * loss_t

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item() * xb.shape[0]

            if cfg.verbose and (epoch % 25 == 0 or epoch == cfg.epochs - 1):
                now = time.perf_counter()
                print(
                    f"[{self.name}] epoch={epoch:04d} "
                    f"loss={total_loss / len(dataset):.4f} "
                    f"since_last={now - last_log_time:.2f}s",
                    flush=True,
                )
                last_log_time = now

        return self

    def predict_potential_outcomes(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")

        X = np.asarray(X, dtype=np.float32)
        Xs = self._standardize_X_transform(X).astype(np.float32)
        device = self._device()

        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(Xs, dtype=torch.float32).to(device)
            mu, _ = self.model(xb)
            mu = mu.cpu().numpy()

        # Restore the original outcome scale.
        return mu * self.y_std_ + self.y_mean_

    def predict_propensity(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")

        X = np.asarray(X, dtype=np.float32)
        Xs = self._standardize_X_transform(X).astype(np.float32)
        device = self._device()

        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(Xs, dtype=torch.float32).to(device)
            _, logits = self.model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs


def make_random_forest(
    random_state: int = 42,
    n_estimators: int = 300,
    min_samples_leaf: int = 20,
    n_jobs: int = -1,
):
    """Create a random-forest regressor with explicit simulation settings."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )

# The MEDUSA network architecture is imported from medusa.py (see the top of
# this module). MedusaConfig and MedusaCausalModel below keep the exact
# training objective used by the validation suite.

@dataclass
class MedusaConfig:
    """Hyperparameters for MEDUSA's architecture, objective, and training."""

    latent_dim: int = 12
    treatment_bits: Optional[int] = None
    epochs: int = 100
    batch_size: int = 256
    lr: float = 8e-4
    weight_decay: float = 2e-4
    alpha_propensity: float = 0.3
    beta_anchor: float = 0.05
    gradient_clip: float = 5.0
    verbose: bool = False
    device: Optional[str] = None
    random_state: int = 42


class MedusaCausalModel(BaseCausalModel):
    """Train MEDUSA for binary outcomes with propensity and anchor penalties.

    ``predict_potential_outcomes(X)`` returns the outcome regressions
    ``mu_t(X)``.
    """

    def __init__(
        self,
        K: int,
        config: Optional[MedusaConfig] = None,
        name: str = "MEDUSA",
    ):
        if K < 2:
            raise ValueError("K must be at least 2.")
        self.K = K
        self.name = name
        self.config = config or MedusaConfig()
        self.model = None
        self.x_mean_ = None
        self.x_std_ = None
        self.treatment_bits_ = None

    def _device(self):
        if self.config.device is not None:
            return self.config.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _standardize_X_fit(self, X):
        self.x_mean_ = X.mean(axis=0, keepdims=True)
        self.x_std_ = X.std(axis=0, keepdims=True) + 1e-8
        return (X - self.x_mean_) / self.x_std_

    def _standardize_X_transform(self, X):
        return (X - self.x_mean_) / self.x_std_

    def _infer_treatment_bits(self):
        return self.config.treatment_bits or int(np.ceil(np.log2(self.K)))

    @staticmethod
    def _indices_to_bits_np(T, n_bits):
        T = np.asarray(T, dtype=np.int64).reshape(-1)
        powers = 2 ** np.arange(n_bits, dtype=np.int64)
        return ((T[:, None] // powers[None, :]) % 2).astype(np.float32)

    @staticmethod
    def _bits_to_indices_np(T_bits):
        T_bits = np.asarray(T_bits, dtype=np.float32)
        powers = 2 ** np.arange(T_bits.shape[1], dtype=np.int64)
        return np.sum(T_bits * powers[None, :], axis=1).astype(np.int64)

    @staticmethod
    def _bits_to_indices_torch(T_bits):
        powers = 2 ** torch.arange(
            T_bits.shape[1], device=T_bits.device, dtype=T_bits.dtype
        )
        return torch.sum(T_bits.float() * powers.view(1, -1), dim=1).long()

    def _as_treatment_bits(self, T):
        T = np.asarray(T)
        if T.ndim == 1:
            indices = T.astype(np.int64).reshape(-1)
            if np.any(indices < 0) or np.any(indices >= self.K):
                raise ValueError(f"T must contain indices from 0 to {self.K - 1}.")
            return self._indices_to_bits_np(indices, self.treatment_bits_)

        bits = T.astype(np.float32)
        if bits.ndim != 2 or bits.shape[1] != self.treatment_bits_:
            raise ValueError(
                f"MEDUSA expected T with {self.treatment_bits_} bits; "
                f"received shape {bits.shape}."
            )
        indices = self._bits_to_indices_np(bits)
        if np.any(indices < 0) or np.any(indices >= self.K):
            raise ValueError("T contains a treatment code with no associated arm.")
        return bits

    @staticmethod
    def _validate_binary_y(Y):
        unique = np.unique(Y)
        if not (len(unique) <= 2 and np.all(np.isin(unique, [0.0, 1.0]))):
            raise ValueError(
                "This version requires binary Y encoded as 0/1; "
                f"received values: {unique.tolist()}."
            )

    def _balanced_factual_loss(self, logits, targets, treatments):
        """Average binary cross-entropy within arms, then across present arms."""
        individual = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        one_hot = F.one_hot(treatments, num_classes=self.K).to(individual.dtype)
        counts = one_hot.sum(dim=0)
        sums = (one_hot * individual.unsqueeze(1)).sum(dim=0)
        valid = counts > 0
        return (sums[valid] / counts[valid]).mean()

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """Fit the factual outcome, propensity, and anchor objectives."""
        cfg = self.config
        np.random.seed(cfg.random_state)
        torch.manual_seed(cfg.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_state)

        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32).reshape(-1)
        self._validate_binary_y(Y)
        if len(X) != len(Y):
            raise ValueError("X and Y must have the same number of rows.")

        self.treatment_bits_ = self._infer_treatment_bits()
        T_bits = self._as_treatment_bits(T)
        if len(T_bits) != len(X):
            raise ValueError("X and T must have the same number of rows.")
        Xs = self._standardize_X_fit(X).astype(np.float32)
        device = self._device()

        self.model = MEDUSA(
            input_size=X.shape[1],
            latent_dim=cfg.latent_dim,
            num_treatments=self.K,
            treatment_bits=self.treatment_bits_,
        ).to(device)

        dataset = TensorDataset(
            torch.tensor(Xs, dtype=torch.float32),
            torch.tensor(T_bits, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        # Divide by log(K) so uniform-propensity cross-entropy is near one.
        propensity_scale = max(float(np.log(self.K)), 1e-8)

        last_log_time = time.perf_counter()
        for epoch in range(cfg.epochs):
            self.model.train()
            total_loss = 0.0
            for xb, tb_bits, yb in loader:
                xb = xb.to(device)
                tb_bits = tb_bits.to(device)
                tb = self._bits_to_indices_torch(tb_bits)
                y_target = yb.to(device).view(-1)

                (
                    factual_out,
                    outputs_all,
                    _,  # The auxiliary classifier output is not in this loss.
                    _,  # The reconstruction output is not in this loss.
                    _,
                    propensity_logits,
                ) = self.model(xb, tb_bits)

                factual_logit = factual_out.view(-1)
                loss_factual = self._balanced_factual_loss(
                    factual_logit, y_target, tb
                )
                loss_propensity = (
                    F.cross_entropy(propensity_logits, tb) / propensity_scale
                )

                mask0 = tb == 0
                if mask0.any():
                    loss_anchor = F.binary_cross_entropy_with_logits(
                        outputs_all[mask0, 0], y_target[mask0]
                    )
                else:
                    loss_anchor = torch.zeros((), device=device)

                loss = (
                    loss_factual
                    + cfg.alpha_propensity * loss_propensity
                    + cfg.beta_anchor * loss_anchor
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Loss became NaN/Inf")

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=cfg.gradient_clip
                )
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            if cfg.verbose and (epoch % 25 == 0 or epoch == cfg.epochs - 1):
                now = time.perf_counter()
                print(
                    f"[{self.name}] epoch={epoch:04d} "
                    f"loss={total_loss / len(dataset):.4f} "
                    f"since_last={now - last_log_time:.2f}s",
                    flush=True,
                )
                last_log_time = now
        return self

    def predict_potential_outcomes(self, X: np.ndarray) -> np.ndarray:
        """Return ``mu_t(X) = P(Y(t)=1 | X)`` without inverse-weighting."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted.")
        X = np.asarray(X, dtype=np.float32)
        Xs = self._standardize_X_transform(X).astype(np.float32)
        device = self._device()
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(Xs, dtype=torch.float32, device=device)
            t0 = torch.zeros(
                (len(Xs), self.treatment_bits_),
                dtype=torch.float32,
                device=device,
            )
            outputs_all = self.model(xb, t0)[1]
            return torch.sigmoid(outputs_all).cpu().numpy()

