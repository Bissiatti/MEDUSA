# Controlled Validation for Multi-Treatment Causal Estimators

A self-contained benchmark suite for evaluating **multi-treatment causal
estimators** — including the deep model **MEDUSA** — on synthetic data with
**known ground-truth potential outcomes**. The suite stress-tests what
happens when parts of the treatment space are unobserved (support removal),
which is exactly the situation causal models face with rare events.

Everything runs from this directory: no external data, no sibling
repositories, no private files.

## What the suite does

1. **Generates** an observational population from a fixed data-generating
   process (DGP): 12 structural covariates + 20 noise covariates, a
   confounded treatment assignment over `2^4 = 16` treatment regimes
   (4 binary treatment bits), a nonlinear potential-outcome surface with
   interactions up to order 4, and Bernoulli factual outcomes.
2. **Computes exact ground truth**: the conditional mean `mu(x, t) =
   E[Y(t) | X = x]` is obtained in closed form by integrating the latent
   Gaussian outcome terms with Gauss-Hermite quadrature, so metrics compare
   models against the true causal surface — not against noisy realizations.
3. **Builds controlled scenarios** that remove treatment support in specific
   ways (see *Scenario families* below), approximating a weak structured
   positivity condition on four covariate strata: baseline arm observed,
   `K` supported nonzero arms, and full rank of the supported treatment
   vectors.
4. **Fits each estimator** on the support-restricted training sample and
   **evaluates** it on a held-out population sample: RMSE against
   `mu(x, t)`, CATE/ATE errors against the baseline arm, optimal-policy
   regret and accuracy, treatment recovery, fit time, and paired contrasts
   between matched scenarios, with bootstrap uncertainty.

### Models included

| Model | Implementation | Notes |
|---|---|---|
| `S-Learner_RF` | Random forest on `[X, one-hot(T)]` | scikit-learn, CPU |
| `T-Learner_RF` | One random forest per arm | Arms missing from training are reported as unsupported |
| `HydraNet` | Shared representation + one outcome head per arm + propensity head | PyTorch |
| `MEDUSA` | Shared outcome network over treatment **bit vectors** + propensity head | PyTorch, see [MEDUSA.md](MEDUSA.md) |

## Repository contents

| File | Purpose |
|---|---|
| `run_validation.py` | The validation runner: DGP, scenario builders, metrics, CLI. |
| `models.py` | Estimator suite used by the runner (S/T-Learner, HydraNet, MEDUSA). |
| `medusa.py` | MEDUSA neural network architecture used by the estimator in `models.py`. See [MEDUSA.md](MEDUSA.md). |
| `config.json` | Default simulation, DGP, and model hyperparameters. |
| `requirements.txt` | Python dependencies. |

## Installation

Requires Python 3.10+ (developed on 3.14). A CUDA GPU accelerates the
neural models but is not required.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

```bash
# Full run with the defaults from config.json (30 replicates, all families,
# all models). This is expensive; see the cheaper options below.
python run_validation.py

# Fast smoke test: build every scenario and print support diagnostics
# without fitting any model.
python run_validation.py --support-only

# A small real run: 2 replicates, one family, one model, on GPU if available.
python run_validation.py --replicates 2 --families ABC --model MEDUSA

# Discover what you can select.
python run_validation.py --list-models
python run_validation.py --list-families
```

Results are written to `<output_dir>/results/` (or
`results_<families>/` for partial runs), where `output_dir` comes from
`config.json` and is resolved relative to this directory. Each run also
saves `resolved_config.json` with the exact settings used.

Useful everyday commands:

```bash
# Regenerate summaries and figures from an existing run without refitting.
python run_validation.py --postprocess-only --outdir results

# Override the config file.
python run_validation.py --config my_config.json

# Watch neural training progress.
python run_validation.py --families ABC --model MEDUSA --verbose-models
```

## CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--config PATH` | `config.json` | JSON file; uses its `base` object. |
| `--replicates N` | from config | Monte-Carlo repetitions (`n_simulations`). |
| `--seed N` | from config | Base seed. |
| `--n N` | from config | Population size; minimum 16384. |
| `--eval-fraction F` | from config | Fraction reserved for population evaluation. |
| `--outdir PATH` | from config | Output directory (overrides the automatic one). |
| `--model NAME` / `--models LIST` | `all` | Restrict models (`S-Learner_RF,T-Learner_RF,HydraNet,MEDUSA`). |
| `--family NAME` / `--families LIST` | `all` | Restrict scenario families (see below). |
| `--list-models` / `--list-families` | — | Print selectable names and exit. |
| `--device {auto,cpu,cuda}` | `auto` | Device for HydraNet/MEDUSA; random forests always use CPU. |
| `--verbose-models` | off | Print epoch/loss logs from the neural models. |
| `--rf-n-jobs N` | `-1` | Random-forest workers. |
| `--max-train-n N` | `8192` | Training-row budget per scenario (minimum 400). |
| `--dominant-region-share F` | `0.70` | Training share of the dominant covariate region in regional families. |
| `--training-outcome {clean,contaminated}` | `clean` | `clean` trains on the pre-contamination Bernoulli label (primary mode); `contaminated` adds observation noise as an extra stress. |
| `--support-only` | off | Build scenarios and print support diagnostics; no fitting. |
| `--postprocess-only` | off | Rebuild `summary.csv` and figures from an existing output directory. |

## Scenario families

Scenarios are grouped into families. Selecting a family automatically adds
the reference families needed for its paired contrasts (`BASE` is always
included). Run `--list-families` for the same table with dependencies.

| Family | What it tests |
|---|---|
| `BASE` | Literal full `2^K \| x` reference: every selected `x` observed under all 16 arms (oracle performance floor). |
| `ABC` | Core comparison: `A_STRONG` (all arms observed everywhere, one factual row per unit), `B_WEAK` (only baseline + canonical basis), `C_RANK3` (rank-deficient support). |
| `D` | Covariate-space extrapolation: same treatment support, but the top covariate-score quartile is removed from training. |
| `E` / `F` / `G` | Regional rank ladders with canonical / general / redundant treatment bases; one covariate region dominates the training share. |
| `H` | Localized four-step support/rank ladder in one dominant region (full 16 arms → rank-4 redundant → rank-4 canonical → rank-3). |
| `I` | Theorem-safe supported-only ladder (evaluation restricted to supported arms). |
| `J` | Rank-4 support smoothing: nested 16/12/8/5-arm supports at fixed rank. |
| `K_NEIGHBOR` | Interleaved nearest-neighbor experiment: can full-support neighbors help extrapolate missing arms? |
| `K_MOD5` | Interleaved modulo-5 supports of rank 4/3/2/1 spread across each region. |
| `K_RANDOM` / `K_RANDOM3` / `K_EXTREME` | Modulo-8 neighbor supports: diverse rank-4 bases, rank-3/rank-2 supports, and maximum-module extremes. |
| `L` | Rank-4 norm-richness experiment (compared against the `H` ladder). |

The full formal description of each scenario is in the module docstring of
`run_validation.py`.

## Configuration (`config.json`)

The runner reads the `base` object. Keys are grouped as follows:

- **Simulation**: `n_simulations`, `seed`, `n`, `test_size`, `models`,
  `output_dir`.
- **DGP**: `n_bits`, `threshold`, `p_noise`, `observation_noise`,
  `latent_noise`, `heteroscedastic_noise`, `treatment_scale`,
  `heterogeneity_scale`, `synergy_scale`, `max_interaction_order`,
  `interaction_density`, `treatment_assignment_strength`,
  `observed_confounding_strength`, `hidden_confounding_strength`, and the
  remaining `dgp`/`split`/`support` controls.
- **Random forests**: `rf_trees`, `rf_min_samples_leaf`.
- **HydraNet**: `hydra_hidden`, `hydra_depth`, `hydra_dropout`,
  `hydra_epochs`, `hydra_batch_size`, `hydra_lr`, `hydra_weight_decay`,
  `hydra_alpha_propensity`, `verbose_hydra`.
- **MEDUSA**: `medusa_latent_dim`, `medusa_epochs`, `medusa_batch_size`,
  `medusa_lr`, `medusa_weight_decay`, `medusa_alpha_propensity`,
  `medusa_beta_anchor`, `medusa_gradient_clip`, `verbose_medusa`.
- **Reporting**: `bootstrap_repetitions`, `save_prediction_diagnostics`,
  `save_individual_ate_errors`, `save_test_predictions`.

CLI flags override the corresponding keys.

## Outputs

Each run writes CSVs and figures to the output directory, including:

- `seed_metrics.csv` — per-replicate metrics for every scenario × model.
- `summary.csv` — metrics aggregated across replicates with bootstrap
  intervals.
- `paired_contrasts.csv` — matched scenario comparisons.
- `relative_worsening_vs_full2k.csv` — deterioration relative to the
  full-support oracle.
- `population_ate_by_treatment.csv`, `population_cate_by_treatment.csv` —
  population-level effect errors per arm.
- `population_policy_class_metrics.csv`,
  `population_treatment_recovery_by_class.csv` — policy evaluation.
- `scenario_catalog.csv`, `local_support_diagnostics.csv` — scenario and
  support diagnostics.
- Per-family metric files (`h_*`, `j_*`, `k_*`, `l_*`, …) and figures
  (`*.png`).
- `resolved_config.json` — the exact configuration of the run.

## Adding your own model

1. Implement the estimator interface in `models.py` (or import it there):

   ```python
   class MyModel:
       name: str
       K: int
       def fit(self, X, T, Y): ...                  # T: arm indices 0..K-1, Y: binary
       def predict_potential_outcomes(self, X): ... # returns array of shape (n, K)
   ```

2. Register a factory in `build_model_factories()` inside
   `run_validation.py` (see the existing `make_medusa` factory for the
   pattern), add the name to `AVAILABLE_MODELS`, and pick a seed offset in
   `MODEL_SEED_OFFSETS`.

The runner handles scenario construction, evaluation, and reporting.

## Reproducibility

Every run is fully seeded (`config.json` `seed`, per-model offsets) and
saves its resolved configuration. Neural models run on GPU when available
(`--device`); note that GPU results can differ slightly across hardware.
The results generated for the accompanying study use the estimator
configuration in `models.py` and the network architecture in `medusa.py`.
