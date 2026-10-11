# MEDUSA

MEDUSA is a PyTorch model for estimating binary potential outcomes across
multiple treatment arms. It encodes covariates once, then shares one outcome
network across all arms. A separate head predicts treatment propensities.

## Files

- `medusa.py` defines the neural network architecture.
- `models.py` defines `MedusaConfig` and `MedusaCausalModel`, including the
  training objective and potential-outcome predictions.
- `run_validation.py` reads the `medusa_*` settings from `config.json` and
  builds the estimator used in the validation suite.

## Architecture

```text
                   +----------------------+
                   |                      v
X (p features) -> Encoder -> z -> Propensity head -> treatment logits
                              |
                              +-> Shared outcome network [z || bits(t)]
                                      |                 |
                                      v                 v
                               factual logit      all-arm logits
```

The encoder is an MLP with widths `p -> 64 -> 32 -> latent_dim`, using layer
normalization and dropout. The propensity head maps the latent representation
to one logit per treatment. The shared outcome network takes the latent
representation concatenated with the treatment's raw binary code and returns
an outcome logit. Applying a sigmoid gives `P(Y(t) = 1 | X)`.

For `K` treatment arms, each arm is encoded with `ceil(log2(K))` bits. Arm zero
uses the all-zero code. The network forward pass returns three tensors:

1. The factual outcome logit, with shape `(batch_size, 1)`.
2. Outcome logits for all arms, with shape `(batch_size, K)`.
3. Treatment propensity logits, with shape `(batch_size, K)`.

## Training objective

For binary outcomes, `MedusaCausalModel.fit` minimizes:

```text
L = L_factual + alpha_propensity * L_propensity + beta_anchor * L_anchor
```

- `L_factual` is binary cross-entropy for the observed treatment. It averages
  the loss within each treatment arm present in the batch, then across arms.
- `L_propensity` is cross-entropy for the observed treatment, divided by
  `log(K)` to keep its scale comparable across treatment counts.
- `L_anchor` is binary cross-entropy for the predicted outcome under arm zero,
  evaluated on observations assigned to arm zero. It is zero for a batch with
  no arm-zero observations.

The estimator uses AdamW, clips the gradient norm, and raises an error if the
training loss becomes non-finite. `predict_potential_outcomes(X)` returns the
sigmoid-transformed all-arm logits as an `(n_samples, K)` array.

## Configuration

`MedusaConfig` is defined in `models.py`. The runner maps these `config.json`
keys to it:

| `config.json` key | Purpose |
|---|---|
| `medusa_latent_dim` | Dimension of the learned representation. |
| `medusa_epochs` | Number of training epochs. |
| `medusa_batch_size` | Number of observations per training batch. |
| `medusa_lr` | AdamW learning rate. |
| `medusa_weight_decay` | AdamW weight decay. |
| `medusa_alpha_propensity` | Weight of the propensity loss. |
| `medusa_beta_anchor` | Weight of the arm-zero anchor loss. |
| `medusa_gradient_clip` | Maximum gradient norm. |
| `verbose_medusa` | Enables periodic epoch and loss messages. |

The runner also sets the treatment-bit count, device, and random seed from the
scenario and command-line settings.

## Raw network example

```python
import torch
from medusa import MEDUSA

net = MEDUSA(input_size=10, latent_dim=8, num_treatments=4, treatment_bits=2)
x = torch.randn(32, 10)
treatment_bits = torch.randint(0, 2, (32, 2)).float()
factual_logits, all_arm_logits, propensity_logits = net(x, treatment_bits)
```
