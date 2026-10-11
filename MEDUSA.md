# MEDUSA — Multi-treatment deep structured representation for counterfactual estimation

MEDUSA is a deep multi-treatment causal model implemented in PyTorch. Given
covariates `X`, a discrete treatment `T` with `K` arms, and a binary outcome
`Y`, it learns the potential-outcome surface `mu_t(X) = P(Y(t) = 1 | X)` for
**every** arm `t` — including arms never observed for similar units — by
sharing one outcome network across all arms and regularizing it with a
propensity head and a doubly robust moment penalty.

Two files in this repository involve MEDUSA:

- **`medusa.py`** — the canonical, self-contained implementation: the
  `MEDUSA` network, `MedusaConfig`, and the full-featured
  `MedusaCausalModel` estimator (with the optional doubly robust moment
  penalty and Hajek-AIPW population estimates). **Use this one** for your own
  data.
- **`models.py`** — the estimator suite used by `run_validation.py`. It
  imports the `MEDUSA` architecture from `medusa.py` and keeps the exact
  training objective configured for the validation study (the doubly robust
  penalty disabled, matching `config.json`).

## Architecture

```text
                X (p covariates)
                 |
            +----v-----+      +-----------+
            | Encoder  +----->|  Decoder  |--> X_hat (auxiliary reconstruction)
            |  p->64->32->d   +-----------+
            +----+-----+
                 | z (latent, dim d)
       +---------+-------------------+
       |                             |
+------v-------+            +--------v---------+
| Propensity   |            | Shared outcome   |
| head         |            | net (per arm t)  |
| d->64->K     |            | [z || bits(t)]   |
+------+-------+            | ->128->64->1     |
       |                    +--------+---------+
       | logits g(X)                 | logits mu_t(X), t = 0..K-1
       v                             v
  P(T=t|X)                    P(Y(t)=1|X)  (after sigmoid)
```

Components:

- **Encoder** `phi(X)`: MLP `p -> 64 (LayerNorm, ReLU, Dropout 0.1) -> 32
  (ReLU, Dropout 0.1) -> d (ReLU)` producing the latent representation `z`.
- **Decoder**: `d -> 32 -> p`, auxiliary reconstruction head (no loss term in
  the current version; kept for architectural compatibility).
- **Propensity head** (DragonNet core): `d -> 64 -> K` logits for
  `P(T = t | X)`.
- **Shared outcome network**: a single `d + b -> 128 -> 64 -> 1` MLP evaluated
  at every treatment arm. Arms are encoded as **raw binary bit vectors**
  `bits(t)` with `b = ceil(log2 K)` bits, concatenated to `z` — no learned
  embedding — so the Hamming geometry between arms is preserved in the input
  space. The baseline arm `0` is the all-zero vector.
- **Classifier** (factual refinement head): `[x || z || bits(t) || mu_t] ->
  64 -> 32 -> 1`. Disabled in the current loss (`delta_class = 0`).

All linear layers use Xavier-uniform initialization.

## Training objective

For binary `Y in {0, 1}`, the per-batch loss in `medusa.py` is:

```text
L = L_factual + alpha * L_propensity + gamma * L_DR + beta * L_anchor
```

- **`L_factual`** — binary cross-entropy of the factual logit, averaged
  **within** each treatment arm present in the batch and then **across** arms
  (balanced factual loss; arms contribute equally regardless of frequency).
- **`L_propensity`** — cross-entropy of the propensity logits against the
  observed arm, divided by `log(K)` so a uniform propensity scores ~1.
  Weighted by `alpha_propensity`.
- **`L_DR`** — squared **normalized AIPW moment** per arm: for each arm `t`,
  the Hajek-normalized inverse-propensity-weighted mean residual
  `sum_i I(T_i=t)/g_t(X_i) * (Y_i - mu_t(X_i)) / sum_i I(T_i=t)/g_t(X_i)`
  (in `[-1, 1]` for binary `Y`), squared and averaged over present arms. The
  propensity is **detached** so the outcome loss cannot reshape `g` just to
  shrink the weights. Weighted by `gamma_dr`; set `gamma_dr = 0` to disable.
- **`L_anchor`** — BCE of the counterfactual head output for the baseline
  arm `t = 0` on units actually observed under `t = 0`. Weighted by
  `beta_anchor`.

Optimization: AdamW with gradient-norm clipping (`gradient_clip`); NaN/Inf
losses raise immediately. The variant in `models.py` used by the validation
suite corresponds to `gamma_dr = 0`.

## Doubly robust population estimates

`MedusaCausalModel.estimate_population_potential_outcomes(X, T, Y)` estimates
`E[Y(t)]` with **Hajek-AIPW** (normalized augmented inverse propensity
weighting): the plugin mean of `mu_t(X)` plus an IPW correction of the
residuals. The estimator is doubly robust — consistent if either the outcome
model or the propensity model is correct — under positivity and
out-of-sample or cross-fitted evaluation.
`estimate_ate(X, T, Y, reference=0)` returns `E[Y(t)] - E[Y(reference)]` for
all arms.

## Quick start

```python
import numpy as np
from medusa import MedusaCausalModel, MedusaConfig

config = MedusaConfig(
    latent_dim=12,
    epochs=100,
    batch_size=256,
    lr=8e-4,
    weight_decay=2e-4,
    alpha_propensity=0.3,
    gamma_dr=0.1,
    beta_anchor=0.05,
    device=None,          # "cuda", "cpu", or None for auto
    random_state=42,
)

model = MedusaCausalModel(K=4, config=config)
model.fit(X, T, Y)                     # X: (n, p), T: (n,) in 0..K-1, Y: (n,) in {0,1}

mu  = model.predict_potential_outcomes(X_new)   # (n, K) probabilities P(Y(t)=1|X)
g   = model.predict_propensity(X_new)           # (n, K) propensities P(T=t|X)
pop = model.estimate_population_potential_outcomes(X_val, T_val, Y_val)  # E[Y(t)]
ate = model.estimate_ate(X_val, T_val, Y_val)   # E[Y(t)] - E[Y(0)]
```

`T` may also be passed directly as bit vectors of shape
`(n, ceil(log2 K))`. `Y` must be binary (`0`/`1`); covariates are
standardized internally.

### Using the raw network

```python
import torch
from medusa import MEDUSA

net = MEDUSA(input_size=10, latent_dim=12, num_treatments=4, treatment_bits=2)
x = torch.randn(32, 10)
t_bits = torch.randint(0, 2, (32, 2)).float()
factual_logit, all_logits, y_logit, x_recon, z, propensity_logits = net(x, t_bits)
```

## Configuration reference (`MedusaConfig`)

| Field | Default | Meaning |
|---|---|---|
| `latent_dim` | 12 | Dimension `d` of the representation `z`. |
| `treatment_bits` | `None` | Bits per arm code; defaults to `ceil(log2 K)`. |
| `epochs` | 100 | Training epochs. |
| `batch_size` | 256 | Minibatch size. |
| `lr` | 8e-4 | AdamW learning rate. |
| `weight_decay` | 2e-4 | AdamW weight decay. |
| `alpha_propensity` | 0.3 | Weight of the propensity cross-entropy. |
| `gamma_dr` | 0.1 | Weight of the normalized DR moment penalty (`0` disables it). |
| `beta_anchor` | 0.05 | Weight of the baseline-arm anchor BCE. |
| `propensity_eps` | 1e-6 | Lower clamp for propensities in IPW terms. |
| `gradient_clip` | 5.0 | Max gradient norm. |
| `device` | `None` | `"cuda"`, `"cpu"`, or `None` (auto-detect). |
| `random_state` | 42 | Seed for NumPy and PyTorch. |

`treatment_embed_dim` and `delta_class` are retained only for backward
compatibility with older configs and have no effect.
