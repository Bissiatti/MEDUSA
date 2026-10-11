"""MEDUSA: multi-treatment causal inference with deep structured representations.

Standalone PyTorch implementation of the MEDUSA architecture for estimating
potential outcomes under multiple discrete treatments. See MEDUSA.md for the
full model documentation.

Contents
--------
- ``MEDUSA``: the raw ``torch.nn.Module`` architecture.
- ``MedusaConfig``: hyperparameters for the architecture and training loop.
- ``MedusaCausalModel``: high-level estimator implementing the shared
  ``fit(X, T, Y)`` / ``predict_potential_outcomes(X)`` interface, plus
  doubly robust population estimates via Hajek-AIPW.

Requires NumPy and PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MEDUSA(nn.Module):
    """Neural network that predicts outcomes and treatment propensities.

    The network encodes covariates ``X`` into a latent representation ``z``,
    predicts treatment propensities from ``z`` (DragonNet-style head), and
    evaluates a shared outcome network at every treatment arm. Treatment arms
    are encoded as raw bit vectors concatenated to ``z`` (no learned
    embedding), which preserves the Hamming geometry between arms.
    """

    def __init__(self, input_size, latent_dim, num_treatments, treatment_bits):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.num_treatments = num_treatments
        self.treatment_bits = treatment_bits

        # Representation phi(X).
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, latent_dim),
            nn.ReLU(),
        )

        # Auxiliary reconstruction decoder.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_size),
        )

        # Treatment-propensity head (DragonNet core).
        self.propensity_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_treatments),
        )

        # Counterfactual outcome network, evaluated at every arm.
        self.shared_net = nn.Sequential(
            nn.Linear(latent_dim + self.treatment_bits, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Final factual refinement head.
        self.classifier = nn.Sequential(
            nn.Linear(input_size + latent_dim + self.treatment_bits + 1, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def _indices_to_bits(self, indices):
        powers = 2 ** torch.arange(self.treatment_bits, device=indices.device)
        return ((indices.view(-1, 1).long() // powers.view(1, -1)) % 2).float()

    def forward(self, x, t_bits):
        # Representation.
        z = self.encoder(x)  # [B, latent]
        x_recon = self.decoder(z)

        # Propensity (DragonNet).
        propensity_logits = self.propensity_head(z)  # [B, T]

        # Counterfactual outcomes.
        t_bits = t_bits.float()
        if t_bits.ndim == 1:
            t_bits = t_bits.view(1, -1)

        factual_inputs = torch.cat([z, t_bits], dim=1)
        factual_logit = self.shared_net(factual_inputs)  # [B, 1]

        all_treatments = torch.arange(self.num_treatments, device=x.device)
        all_bits = self._indices_to_bits(all_treatments)  # [T, treatment_bits]

        z_expanded = z.unsqueeze(1).repeat(1, self.num_treatments, 1)
        all_bits_expanded = all_bits.unsqueeze(0).repeat(x.size(0), 1, 1)

        inputs = torch.cat([z_expanded, all_bits_expanded], dim=-1)
        outputs = self.shared_net(inputs).squeeze(-1)  # [B, T] logits

        # Final factual classifier.
        enhanced = torch.cat([x, z, t_bits, factual_logit], dim=1)
        y_logit = self.classifier(enhanced)  # [B, 1]

        return factual_logit, outputs, y_logit, x_recon, z, propensity_logits


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
    gamma_dr: float = 0.1
    beta_anchor: float = 0.05
    propensity_eps: float = 1e-6
    gradient_clip: float = 5.0
    verbose: bool = False
    device: Optional[str] = None
    random_state: int = 42

    # Kept only for backward compatibility with older configs; both are
    # disabled and do not enter the loss.
    treatment_embed_dim: int = 0
    delta_class: float = 0.0


class MedusaCausalModel:
    """Train MEDUSA for binary outcomes with a normalized DR moment penalty.

    ``predict_potential_outcomes(X)`` returns the outcome regressions
    ``mu_t(X)``.

    ``estimate_population_potential_outcomes(X, T, Y)`` returns Hajek-AIPW
    population estimates. The doubly robust property belongs to that
    estimator: it is consistent if either the outcome model or the propensity
    model is correct, under positivity and out-of-sample/cross-fitted
    evaluation.
    """

    def __init__(self, K: int, config: Optional[MedusaConfig] = None, name: str = "MEDUSA-DR"):
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
        return self.config.treatment_bits or int(math.ceil(math.log2(self.K)))

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

    def _normalized_dr_moment_loss(self, mu_all, propensity, treatments, targets):
        """Squared normalized AIPW moment, averaged across arms.

        For each treatment t::

            m_t = sum_i I(T_i=t)/g_t(X_i) * (Y_i - mu_t(X_i))
                  ------------------------------------------------
                         sum_i I(T_i=t)/g_t(X_i)

        Each m_t lies in [-1, 1] for binary Y, so the mean of m_t^2 lies in
        [0, 1] regardless of the number K of treatments.

        The propensity is detached from the graph so the outcome loss cannot
        reshape g just to shrink the weights; g is still trained by
        cross-entropy.
        """
        propensity_safe = propensity.detach().clamp_min(self.config.propensity_eps)
        one_hot = F.one_hot(treatments, num_classes=self.K).to(mu_all.dtype)
        weights = one_hot / propensity_safe
        residuals = targets.unsqueeze(1) - mu_all

        numerator = (weights * residuals).sum(dim=0)
        denominator = weights.sum(dim=0)
        valid = one_hot.sum(dim=0) > 0
        moments = numerator[valid] / denominator[valid].clamp_min(
            self.config.propensity_eps
        )
        return moments.pow(2).mean()

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """Fit the factual, propensity, DR-moment, and anchor objectives."""
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
        propensity_scale = max(float(math.log(self.K)), 1e-8)

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
                    _,  # Auxiliary classifier output: disabled (delta_class=0).
                    _,  # Reconstruction output: no loss in this version.
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

                propensity = torch.softmax(propensity_logits, dim=1)
                mu_all = torch.sigmoid(outputs_all)
                loss_dr = self._normalized_dr_moment_loss(
                    mu_all, propensity, tb, y_target
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
                    + cfg.gamma_dr * loss_dr
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
                print(
                    f"[{self.name}] epoch={epoch:04d} "
                    f"loss={total_loss / len(dataset):.4f}"
                )
        return self

    def predict_potential_outcomes(self, X: np.ndarray) -> np.ndarray:
        """Return ``mu_t(X) = P(Y(t)=1 | X)`` without inverse weighting."""
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

    def predict_propensity(self, X: np.ndarray) -> np.ndarray:
        """Return the treatment propensity matrix ``g_t(X)`` with shape (n, K)."""
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
            propensity_logits = self.model(xb, t0)[5]
            return torch.softmax(propensity_logits, dim=1).cpu().numpy()

    def estimate_population_potential_outcomes(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        return_diagnostics: bool = False,
    ):
        """Estimate E[Y(t)] with normalized Hajek-AIPW for t = 0, ..., K-1.

        For valid DR inference, ``X, T, Y`` must come from a sample that was
        not used to fit the same nuisance functions, or the predictions must
        come from cross-fitting.
        """
        Y = np.asarray(Y, dtype=np.float64).reshape(-1)
        self._validate_binary_y(Y)
        T_array = np.asarray(T)
        if T_array.ndim == 1:
            treatment_indices = T_array.astype(np.int64).reshape(-1)
        else:
            treatment_indices = self._bits_to_indices_np(T_array)
        if not (len(X) == len(treatment_indices) == len(Y)):
            raise ValueError("X, T, and Y must have the same number of rows.")
        if np.any(treatment_indices < 0) or np.any(treatment_indices >= self.K):
            raise ValueError(f"T must contain treatments from 0 to {self.K - 1}.")

        mu = self.predict_potential_outcomes(X).astype(np.float64)
        propensity = self.predict_propensity(X).astype(np.float64)
        propensity = np.maximum(propensity, self.config.propensity_eps)
        one_hot = np.eye(self.K, dtype=np.float64)[treatment_indices]
        weights = one_hot / propensity
        counts = one_hot.sum(axis=0)
        if np.any(counts == 0):
            missing = np.flatnonzero(counts == 0).tolist()
            raise ValueError(
                "AIPW cannot identify treatments missing from the estimation "
                f"sample: {missing}."
            )

        residuals = Y[:, None] - mu
        correction = (weights * residuals).sum(axis=0) / weights.sum(axis=0)
        estimates_raw = mu.mean(axis=0) + correction
        estimates = np.clip(estimates_raw, 0.0, 1.0)

        if not return_diagnostics:
            return estimates

        effective_sample_size = weights.sum(axis=0) ** 2 / np.maximum(
            (weights**2).sum(axis=0), self.config.propensity_eps
        )
        diagnostics = {
            "counts": counts.astype(np.int64),
            "effective_sample_size": effective_sample_size,
            "plugin_estimates": mu.mean(axis=0),
            "dr_correction": correction,
            "raw_estimates_before_probability_clip": estimates_raw,
            "min_observed_propensity": np.array(
                [
                    propensity[treatment_indices == t, t].min()
                    for t in range(self.K)
                ]
            ),
        }
        return estimates, diagnostics

    def estimate_ate(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        reference: int = 0,
    ) -> np.ndarray:
        """Return ``E[Y(t)] - E[Y(reference)]`` using Hajek-AIPW."""
        if reference < 0 or reference >= self.K:
            raise ValueError(f"reference must be between 0 and {self.K - 1}.")
        means = self.estimate_population_potential_outcomes(X, T, Y)
        return means - means[reference]
