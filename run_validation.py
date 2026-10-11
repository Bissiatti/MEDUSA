#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_validation.py

Model-based controlled validation aligned with the formal weak structured
positivity discussion for MEDUSA.

The benchmark keeps the base DGP structure for X, treatment assignment T, the
nonlinear potential-outcome surface, latent outcome noise, Bernoulli factual
outcome, and heteroscedastic label contamination. The controlled scenarios
modify only the *observed training support* through rejection/subsampling from
the generated observational population.

Formal target
-------------
The primary causal estimand is

    mu(x,t) = E[Y(t) | X=x]
    tau_t(x) = mu(x,t) - mu(x,0)
    ATE_t = E_X[tau_t(X)]

For the source DGP, Ypot stores a realization-specific latent probability that
contains Gaussian terms not predictable from X. Therefore this script also
computes the model-based conditional mean mu(x,t) by integrating those Gaussian
latent outcome terms with Gauss-Hermite quadrature. All primary CATE, ATE, and
policy metrics use this conditional mean. The realization-specific Ypot target
is retained only as a secondary stress metric.

Weak structured positivity approximation
-----------------------------------------
The formal local condition is approximated on four pre-specified X strata
(quartiles of the source DGP's S_extra score). In every stratum where the weak
condition is intended to hold, the training sample contains:

  * baseline 0000 with positive empirical mass;
  * K=4 supported nonzero treatment vectors with positive empirical mass;
  * rank(U)=4, where U has those treatment vectors as columns.

The condition is geometric only. It does NOT claim that unsupported mu(x,t) is
nonparametrically identified or linear in t. Unsupported vertices are evaluated
as model-based predictions, which is exactly the intended validation target.

Scenarios
---------
BASE_FULL_2K_GIVEN_X
    Literal fully-crossed reference design. A stratified subset of training
    covariate vectors is selected so that, after crossing each selected x with
    all 2^K=16 actions, the expanded design stays within the total training-row
    budget. Outcomes are generated from the same DGP potential-outcome
    probabilities. Thus, every selected x has all 16 actions:
        |S_train(x)| = 2^K.
    This is an oracle/full-support reference used only as a performance floor
    for reporting percentage deterioration under support removal.

A_STRONG
    All 16 treatment vertices are represented in every X stratum, but—as in an
    ordinary observational sample—each unit contributes only its factual action.

B_WEAK
    Only {0000,e1,e2,e3,e4} is retained in every X stratum. Baseline overlap
    and rank(U)=4 hold, while most joint vertices are unsupported.

C_RANK3
    Only {0000,e1,e2,e3} is retained in every X stratum. Baseline remains
    supported, but rank(U)=3<K.

D_REF_X / D_POOR_X
    Both use the same full-rank canonical treatment generating set. D_REF_X
    retains training observations throughout X, while D_POOR_X removes the
    highest S_extra quartile from training. Both are evaluated on that same
    high-X target region. This separates treatment-space coverage from X-space
    extrapolation.

Regional rank-dominance families
    Four physical X strata Q1..Q4 are assigned local ranks 1..4. The mapping is
    cyclically counterbalanced across successful Monte-Carlo repetitions, so a
    particular rank is not permanently tied to one part of X-space. Four tests
    then make the rank-4, rank-3, rank-2, or rank-1 stratum contribute the
    dominant training share (default 70%; the other three receive 10% each).

    E_CANONICAL_*:
        origin + canonical vectors e1..er.

    F_GENERAL_BASIS_*:
        origin + r arbitrary independent binary vectors; same support
        cardinality as the canonical family.

    G_GENERAL_REDUNDANT_*:
        origin + an arbitrary rank-r set that may contain more nonzero vectors
        without increasing rank. For rank 3, for example, four nonzero vectors
        of rank 3 are allowed, matching constructions such as
        {1101,1001,1000,0101} when feasible in the generated data.

H_LOCAL_SUPPORT_LADDER_*:
    A localized four-step support ladder designed to isolate the information
    lost in one dominant X region (default 70% of training observations) while
    holding the other three X regions at full 16-vertex support.  The same
    dominant physical X region and the same 70/10/10/10 allocation are used in
    all four scenarios within a seed.  The dominant region moves through:

      1. FULL16: all 2^K=16 vertices observed (rank 4);
      2. R4_REDUNDANT: the canonical rank-4 basis {e1,e2,e3,e4} plus one
         additional redundant vertex v (origin + 5 nonzero vertices; 6 total);
      3. R4_CANONICAL: the same canonical basis without v
         {0000,e1,e2,e3,e4} (rank 4; 5 vertices total);
      4. R3_REDUNDANT: the same support as R4_CANONICAL except that one
         canonical direction e_j is replaced by that same redundant vertex v.
         Hence it has 5 vertices total but rank 3.

    Thus R4_CANONICAL and R3_REDUNDANT have exactly the same support
    cardinality and differ by a one-for-one swap: one independent direction is
    removed and replaced by a vertex lying in the span of the three retained
    canonical directions.  This makes the final contrast a matched rank test,
    rather than a comparison of unrelated support orientations.  The omitted
    canonical direction is counterbalanced across successful Monte-Carlo
    repetitions, as is the dominant physical X region.

J_RANK4_SUPPORT_SMOOTHING_*:
    A nested support-cardinality ladder that smooths the large H transition
    from full 16-vertex support to the minimal canonical rank-4 support.  It
    keeps the same dominant X region, the same training-row budget, and full
    16-vertex support in the other three X regions.  Rank remains exactly K=4
    at every rung because {e1,e2,e3,e4} is retained throughout.

    There are only 15 nonzero vertices when K=4, so the mathematically
    consistent total-support ladder is:

      1. FULL16: origin + all 15 nonzero vertices = 16 total;
      2. SUPPORT12: origin + canonical basis + 7 joint vertices = 12 total;
      3. SUPPORT8: origin + canonical basis + 3 of those same joint vertices
         = 8 total;
      4. CANONICAL: origin + the 4 canonical basis vectors = 5 total.

    Thus the user's requested endpoint of "4 vectors plus the origin" is the
    canonical 5-vertex support.  SUPPORT8 is nested in SUPPORT12, which is
    nested in FULL16.  The extra joint vertices are selected without using
    outcomes and are re-randomized across Monte-Carlo repetitions.  The first
    and final rungs reuse the already existing H FULL16 and H R4_CANONICAL
    scenarios exactly, so J adds only the two intermediate model fits.

K_NEIGHBOR_SUPPORT_EXTRAPOLATION_*:
    A nearest-neighbor/interleaved-X experiment built directly from the literal
    full-2^K|x baseline anchors.  Within each physical X region, the selected
    anchors are ordered by S_extra and split by alternating parity.  Consecutive
    anchors therefore form local pairs distributed across the whole region.

    Two matched scenarios use exactly the same X anchors and exactly the same
    realized outcome rows whenever a row is retained:

      1. K_NEIGHBOR_FULL16_REFERENCE: every anchor keeps all 16 treatments;
      2. K_NEIGHBOR_HALF_CANONICAL: odd/even interleaving is preserved, but
         one half of the ordered X anchors keeps only {0,e1,e2,e3,e4}; the
         neighboring half keeps all 16 treatments.

    The evaluation target is the same weak-half X anchors in both scenarios and
    asks for all 16 potential outcomes.  Thus the second scenario tests whether
    full-support X neighbors can help extrapolate the 11 noncanonical treatment
    vertices that were removed at the target X values, without introducing a
    large covariate-region shift.  The full/weak halves are equally represented
    inside every Q1..Q4 stratum and are interleaved across S_extra.

K_UNSUPPORTED_MOD5_*:
    A five-way interleaved extension of K.  Inside every physical X region, the
    literal full-2^K anchors are ordered by S_extra and assigned by ordered
    position modulo 5.  The support profile is therefore spread uniformly
    throughout each region rather than concentrated in a covariate tail:

      position mod 5 = 0: FULL16, all 16 treatment vertices, rank 4;
      position mod 5 = 1: canonical weak support {0,e1,e2,e3,e4}, rank 4;
      position mod 5 = 2: {0,e1,e2,e3}, rank 3;
      position mod 5 = 3: {0,e1,e2}, rank 2;
      position mod 5 = 4: {0,e1}, rank 1.

    K_UNSUPPORTED_FULL16_REFERENCE retains all 16 rows at every selected anchor.
    K_UNSUPPORTED_MOD5 deletes rows according to the profile above, reusing the
    exact same X,T,Y rows from the literal baseline whenever a row is retained.
    Evaluation asks for all 16 potential outcomes separately in each modulo-5
    X population.  Thus rank-4/3/2/1 groups test progressively unsupported
    treatment-space extrapolation while each group remains interleaved across
    the same X regions and has nearby full-support representatives.

K_RANDOM_MOD8_*:
    A heterogeneous rank-4 neighbor-support experiment.  Eight distinct
    five-vertex supports are selected combinatorially.  Every support contains
    the origin plus four linearly independent nonzero treatment vectors, hence
    has total cardinality 5 and rank 4.  Because K=4 gives only 15 nonzero
    vertices, eight such supports cannot be pairwise disjoint: they require
    32 nonzero slots.  The selector therefore solves the strongest feasible
    diversity problem lexicographically: (i) every chosen basis attains the
    maximum possible within-support Hamming diversity; (ii) the union covers
    all 15 nonzero vertices; (iii) no pair of bases shares more than one
    nonzero vertex; and (iv) nonzero-vertex usage is as balanced as possible
    (13 vertices used twice and 2 used three times).

    Within every physical X region, literal full-2^K anchors are ordered by
    S_extra, trimmed to a multiple of eight, and assigned by ordered position
    modulo 8.  A seeded permutation counterbalances which of the eight diverse
    rank-4 supports is attached to each modulo class across replications.
    K_RANDOM_FULL16_REFERENCE keeps all 16 treatment rows at the same anchors,
    whereas K_RANDOM_RANK4_MOD8 retains only the assigned five-vertex support
    at each X.  All 16 potential outcomes are evaluated in both scenarios.
    Thus K-RANDOM tests whether a shared model can borrow complementary
    treatment information across nearby X values when every local support is
    minimal full-rank but the neighborhood collectively spans all 16 vertices.

K_RANDOM3_MOD8_*:
    A stricter neighbor-support experiment paired to K-RANDOM.  It reuses the
    exact same ordered-X modulo-8 anchor partition as K_RANDOM, but no target X
    in the mixed training design has rank 4.  The eight local supports alternate
    between four rank-3 size-4 supports (origin + three independent nonzero
    vectors) and four rank-2 size-3 supports (origin + two independent nonzero
    vectors):

        mod 0,2,4,6: rank 3, |S|=4;
        mod 1,3,5,7: rank 2, |S|=3.

    The eight supports are selected to maximize treatment-vector diversity
    subject to collective full-16 coverage.  For K=4, the selected design
    covers all 15 nonzero vertices, uses each nonzero vertex only once or twice,
    has pairwise nonzero overlap at most one, and attains total within-support
    Hamming diversity 95 out of the unconstrained upper bound 96.  The latter
    bound cannot attain full 15-vertex coverage, so 95 is optimal under the
    coverage constraint.

    K_RANDOM_FULL16_REFERENCE is reused as the matched reference, avoiding an
    unnecessary duplicate fit.  K_RANDOM3_RANK32_MOD8 deletes rows from that
    exact reference according to the alternating rank-3/rank-2 supports.  All
    16 potential outcomes are evaluated.  Thus K-RANDOM3 asks whether nearby X
    profiles with complementary but individually rank-deficient support can
    collectively supply enough information for treatment-space extrapolation.

K_EXTREME_*:
    Extreme heterogeneous-neighbor variants of K-RANDOM.  They reuse the exact
    same ordered-X anchors and matched full16 reference as K-RANDOM, but maximize
    the number of distinct local support matrices used across those anchors.

    K_EXTREME_RANK4_MAXMOD enumerates every minimal rank-4 support available for
    K=4: origin plus four linearly independent nonzero vertices.  There are 940
    such supports.  The supports are seed-permuted and assigned sequentially to
    the ordered X anchors, so nearby X values encounter as many distinct full-rank
    orientations as the finite sample permits.

    K_EXTREME_LT4_MAXMOD analogously enumerates every minimal rank-deficient
    support of ranks 1, 2, and 3: origin plus r independent nonzero vertices.
    For K=4 there are 15 rank-1, 105 rank-2, and 430 rank-3 supports, totaling
    550 distinct local matrices.  No local support has rank 4, yet the catalog
    collectively covers all 16 treatment vertices.

    Both variants evaluate all 16 potential outcomes.  The reference fit is not
    duplicated: K_RANDOM_FULL16_REFERENCE is reused because the X anchors are
    exactly the same.  If a user reduces the training budget so far that fewer
    anchors are available than the theoretical catalog size, the effective
    module is the largest feasible value, namely the number of available anchors.

L_RANK4_NORM_RICHNESS_*:
    A four-region rank-4, size-5 benchmark for the combinatorial composition of
    a minimal full-rank support.  Every X region contains the origin plus four
    linearly independent nonzero treatments, so all local supports have rank 4
    and cardinality 5.  The four support profiles differ only in the Hamming
    weights ||t||_0 represented among those four basis vectors:

      1. NORM1:    weights (1,1,1,1), the canonical singleton basis;
      2. NORM12:   weights (1,1,2,2);
      3. NORM123:  weights (1,2,2,3);
      4. NORM1234: weights (1,2,3,4).

    The profile-to-physical-region mapping is cyclically counterbalanced across
    Monte-Carlo repetitions.  Four matched scenarios then make each profile's
    region contribute the dominant share of selected X anchors (default 70%),
    with the other three regions sharing the remainder as in H.  Each selected
    X anchor is crossed only with its region's five supported treatments, giving
    equal representation to the five local vertices and avoiding confounding the
    norm comparison with factual treatment rarity.  All 16 treatment combinations
    are evaluated in every scenario.  L therefore tests whether, conditional on
    the same minimal full-rank geometry, direct observation of progressively richer
    joint-treatment combinations improves model-based prediction of the full
    treatment response surface.

I_THEOREM_SAFE_SUPPORT_LADDER_*:
    A theorem-facing supported-only ladder.  Unlike H, there is no dominant-X
    extrapolation and no evaluation at unsupported treatment actions.  All four
    physical X regions are represented equally in training and in the common
    evaluation set, and every region receives the same treatment-support profile
    within a scenario.  The four matched profiles are:

      1. FULL16: all 16 treatment vertices;
      2. R4_REDUNDANT: {0,e1,e2,e3,e4,v}, rank 4;
      3. R4_CANONICAL: {0,e1,e2,e3,e4}, rank 4;
      4. R3_REDUNDANT: {0,R,v}, rank 3, where R contains the three retained
         canonical directions and v lies in span(R).

    Crucially, the evaluation treatment matrix is reduced together with the
    support: a scenario is evaluated only on treatments that are supported in
    every X region under that scenario.  In addition, all four profiles are
    compared on a fixed common core {0,R}, which is supported everywhere in all
    four profiles.  Thus the I family never asks the model to predict a treatment
    that is absent from the local support being evaluated.  R3_REDUNDANT is a
    boundary control outside the full-rank assumption; no theorem-based claim is
    made for unsupported directions because they are never evaluated.

Most scenarios use the same fixed empirical population-evaluation set within a
seed for population ATE, CATE and policy metrics.  K is intentionally different:
it evaluates the same interleaved weak-X anchors in its two matched scenarios.

Relative deterioration
----------------------
For each model and Monte-Carlo seed, all theory-facing metrics are additionally
reported relative to BASE_FULL_2K_GIVEN_X. For lower-is-better errors,

    worsening_pct = 100 * (metric_scenario - metric_base) / metric_base.

For higher-is-better quantities such as policy accuracy/value,

    worsening_pct = 100 * (metric_base - metric_scenario) / metric_base.

The fully-crossed baseline is intentionally stronger than ordinary observational
support: it is the literal 2^K | x reference requested for measuring information
loss as treatment support is removed.

Primary metrics
---------------
  population_mean_ate_abs_error
  population_cate_rmse_origin
  population_policy_regret
  population_policy_accuracy
  population_treatment_recovery_accuracy
  population_treatment_recovery_hamming_distance
  target_rmse_mu_all
  target_rmse_mu_supported_local
  target_rmse_mu_unsupported_local

The population policy optimum is argmax_t mu(x,t), not argmax of the
realization-specific noisy Ypot.

The sample size is read from ``config.json`` and may be overridden with
``--n``.  Any integer ``n >= 2^14 = 16,384`` is accepted.  This lower bound
preserves enough observational population for the four X strata, all 16
factual arms per stratum in ``A_STRONG``, the canonical ranks, and the regional
rank-dominance families.

Estimators
----------
The benchmark uses the four high-level estimators exported by ``models.py``:
S-Learner RF, T-Learner RF, HydraNet, and MEDUSA.  The treatment-bit design
used here is converted to the categorical treatment indices expected by those
estimators.  HydraNet and MEDUSA use CUDA automatically when it is available.
Because ``TLearnerMulti`` requires observations from every treatment arm, its
results are available only for the fully supported scenarios; structurally
unsupported scenarios are recorded explicitly in ``failures.csv``.

Examples
--------
python3 run_validation.py --list-models
python3 run_validation.py --list-families
python3 run_validation.py --model MEDUSA --device auto
python3 run_validation.py --family L --model MEDUSA --device auto
python3 run_validation.py --models HydraNet,MEDUSA --device cuda
python3 run_validation.py --support-only

By default, simulation and model hyperparameters are read from
``config.json`` next to this script.

Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib, torch
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.spatial.distance import cdist
from scipy.stats import norm, rankdata, t as student_t
import matplotlib.pyplot as plt


def _load_sibling_models_module():
    """Load the sibling models.py file, regardless of the current directory."""
    models_path = Path(__file__).resolve().with_name("models.py")
    module_name = "controlled_validation_models"
    spec = importlib.util.spec_from_file_location(module_name, models_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load models module from {models_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


MODELS_MODULE = _load_sibling_models_module()
SLearnerMulti = MODELS_MODULE.SLearnerMulti
TLearnerMulti = MODELS_MODULE.TLearnerMulti
HydraNetCausalModel = MODELS_MODULE.HydraNetCausalModel
HydraNetConfig = MODELS_MODULE.HydraNetConfig
MedusaCausalModel = MODELS_MODULE.MedusaCausalModel
MedusaConfig = MODELS_MODULE.MedusaConfig
make_random_forest = MODELS_MODULE.make_random_forest


# =====================================================================
# Constants
# =====================================================================

MIN_N = 2**14
N = MIN_N
K_BITS = 4
N_REGIMES = 2**K_BITS
P_STRUCT = 12
P_NOISE = 20
BASE_SEED = 2002

POP_EVAL_FRACTION = 0.20
MAX_TRAIN_N_DEFAULT = 8192
DOMINANT_SHARE_DEFAULT = 0.70
MIN_BASIS_EMPIRICAL_SHARE = 0.03
EPS = 1e-8
FULL2K_BASELINE_NAME = "BASE_FULL_2K_GIVEN_X"

# Base DGP constants
ASSIGNMENT_THRESHOLD = -1.0
ASSIGNMENT_STRENGTH = 1.10
OBSERVED_CONFOUNDING_STRENGTH = 1.0
HIDDEN_CONFOUNDING_STRENGTH = 0.0
ERROR_CORRELATION = 0.08

TREATMENT_SCALE = 0.80
HETEROGENEITY_SCALE = 1.10
INTERACTION_DENSITY = 0.35
SYNERGY_SCALE = 0.70
MAX_INTERACTION_ORDER = 4

LATENT_NOISE_SD = 0.35
REGIME_LATENT_NOISE_SD = 0.30 * LATENT_NOISE_SD
OUTCOME_TEMPERATURE = 1.0
OBSERVATION_NOISE = 0.08
HETEROSCEDASTIC_NOISE = 1.0
GH_NODES = 24

# Physical covariate strata. They are not named by rank.
X_REGIONS = ("X_Q1", "X_Q2", "X_Q3", "X_Q4")

DEFAULT_INPUT_PATH = Path(__file__).resolve().parent / "config.json"
DEFAULT_RESULTS_SUBDIR = "results"
AVAILABLE_MODELS = ("S-Learner_RF", "T-Learner_RF", "HydraNet", "MEDUSA")
# RF parallelism remains enabled on every Python version.  ``models.py``
# handles the Python 3.14/scikit-learn warning-context race inside the threaded
# RandomForest calls without serializing tree construction.
RF_N_JOBS_DEFAULT = -1
MODEL_SEED_OFFSETS = {
    "S-Learner_RF": 0,
    "T-Learner_RF": 1_000,
    "HydraNet": 2_000,
    "MEDUSA": 3_000,
}

# Public CLI names for the controlled test families.  ``ABC`` is the joint
# core family because A, B, and C form one matched comparison.  The K variants
# are exposed separately because their computational costs differ markedly.
AVAILABLE_FAMILIES = (
    "BASE",
    "ABC",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K_NEIGHBOR",
    "K_MOD5",
    "K_RANDOM",
    "K_RANDOM3",
    "K_EXTREME",
    "K",
    "L",
)

FAMILY_DESCRIPTIONS = {
    "BASE": "literal full 2^K|x reference",
    "ABC": "core strong/weak/rank-deficient comparison",
    "D": "covariate-support extrapolation",
    "E": "canonical regional rank ladder",
    "F": "general-basis regional rank ladder (compared with E)",
    "G": "redundant regional rank ladder (with E and F references)",
    "H": "localized support/rank ladder",
    "I": "theorem-safe supported-only ladder",
    "J": "rank-4 support smoothing (uses H endpoints)",
    "K_NEIGHBOR": "interleaved nearest-neighbor support experiment",
    "K_MOD5": "interleaved modulo-5 rank experiment",
    "K_RANDOM": "diverse rank-4 modulo-8 supports",
    "K_RANDOM3": "rank-3/rank-2 modulo-8 supports (compared with K_RANDOM)",
    "K_EXTREME": "maximum-module K variants (compared with K_RANDOM/K_RANDOM3)",
    "K": "all K experiments",
    "L": "rank-4 norm-richness experiment (with H comparison ladder)",
}

# Direct comparison dependencies.  BASE is added to every partial run so the
# standard relative-to-full-2^K metrics remain defined.  Dependencies are
# closed transitively by ``expand_family_dependencies``.
FAMILY_DEPENDENCIES = {
    "F": ("E",),
    "G": ("E", "F"),
    "J": ("H",),
    "L": ("H",),
    "K_RANDOM3": ("K_RANDOM",),
    "K_EXTREME": ("K_RANDOM", "K_RANDOM3"),
}

K_SUBFAMILIES = (
    "K_NEIGHBOR",
    "K_MOD5",
    "K_RANDOM",
    "K_RANDOM3",
    "K_EXTREME",
)


def load_input_defaults(path: Path) -> Dict[str, Any]:
    """Load base settings from a nested or flat JSON configuration."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a JSON object: {path}")
    base = payload.get("base", payload)
    if not isinstance(base, dict):
        raise ValueError(f"The 'base' configuration must be a JSON object: {path}")
    return dict(base)


def configure_dgp_from_input(config: Mapping[str, Any]) -> None:
    """Apply the configuration fields used by this controlled DGP.

    The support scenarios remain specific to this script.  Fields for the
    unrelated suite grids, holdout modes, and artificial support dropping are
    intentionally not applied here.
    """
    global N, BASE_SEED, POP_EVAL_FRACTION, P_NOISE
    global ASSIGNMENT_THRESHOLD, ASSIGNMENT_STRENGTH
    global OBSERVED_CONFOUNDING_STRENGTH, HIDDEN_CONFOUNDING_STRENGTH
    global ERROR_CORRELATION, TREATMENT_SCALE, HETEROGENEITY_SCALE
    global INTERACTION_DENSITY, SYNERGY_SCALE, MAX_INTERACTION_ORDER
    global LATENT_NOISE_SD, REGIME_LATENT_NOISE_SD, OUTCOME_TEMPERATURE
    global OBSERVATION_NOISE, HETEROSCEDASTIC_NOISE

    n_bits = int(config.get("n_bits", K_BITS))
    if n_bits != K_BITS:
        raise ValueError(
            f"This controlled design requires n_bits={K_BITS}; received {n_bits}."
        )
    outcome = str(config.get("outcome", "bernoulli"))
    if outcome != "bernoulli":
        raise ValueError(
            "This controlled validation (and MEDUSA in models.py) requires "
            f"outcome='bernoulli'; received {outcome!r}."
        )

    N = int(config.get("n", N))
    BASE_SEED = int(config.get("seed", BASE_SEED))
    POP_EVAL_FRACTION = float(config.get("test_size", POP_EVAL_FRACTION))
    P_NOISE = int(config.get("p_noise", P_NOISE))

    ASSIGNMENT_THRESHOLD = float(config.get("threshold", ASSIGNMENT_THRESHOLD))
    ASSIGNMENT_STRENGTH = float(
        config.get("treatment_assignment_strength", ASSIGNMENT_STRENGTH)
    )
    OBSERVED_CONFOUNDING_STRENGTH = float(
        config.get("observed_confounding_strength", OBSERVED_CONFOUNDING_STRENGTH)
    )
    HIDDEN_CONFOUNDING_STRENGTH = float(
        config.get("hidden_confounding_strength", HIDDEN_CONFOUNDING_STRENGTH)
    )
    ERROR_CORRELATION = float(config.get("treatment_correlation", ERROR_CORRELATION))
    TREATMENT_SCALE = float(config.get("treatment_scale", TREATMENT_SCALE))
    HETEROGENEITY_SCALE = float(config.get("heterogeneity_scale", HETEROGENEITY_SCALE))
    INTERACTION_DENSITY = float(config.get("interaction_density", INTERACTION_DENSITY))
    SYNERGY_SCALE = float(config.get("synergy_scale", SYNERGY_SCALE))
    MAX_INTERACTION_ORDER = int(
        config.get("max_interaction_order", MAX_INTERACTION_ORDER)
    )
    LATENT_NOISE_SD = float(config.get("latent_noise", LATENT_NOISE_SD))
    REGIME_LATENT_NOISE_SD = 0.30 * LATENT_NOISE_SD
    OUTCOME_TEMPERATURE = float(config.get("outcome_temperature", OUTCOME_TEMPERATURE))
    OBSERVATION_NOISE = float(config.get("observation_noise", OBSERVATION_NOISE))
    HETEROSCEDASTIC_NOISE = float(
        config.get("heteroscedastic_noise", HETEROSCEDASTIC_NOISE)
    )

    if not 0.0 < POP_EVAL_FRACTION < 1.0:
        raise ValueError("test_size must be strictly between 0 and 1.")
    if N < MIN_N:
        raise ValueError(
            f"n must be at least 2^14={MIN_N} to preserve every controlled "
            f"validation scenario; received n={N}. Use, for example, "
            "--n 65000 or --n 65536."
        )
    if P_NOISE < 0:
        raise ValueError("p_noise cannot be negative.")
    if not 1 <= MAX_INTERACTION_ORDER <= K_BITS:
        raise ValueError(f"max_interaction_order must be between 1 and {K_BITS}.")
    if not 0.0 <= INTERACTION_DENSITY <= 1.0:
        raise ValueError("interaction_density must be between 0 and 1.")
    if not (-1.0 / (K_BITS - 1) < ERROR_CORRELATION < 1.0):
        raise ValueError("treatment_correlation does not define a positive covariance matrix.")


def normalize_model_selection(value: str | None) -> Tuple[str, ...]:
    """Return canonical model names from a comma-separated CLI/config value."""
    raw = "all" if value is None else str(value).strip()
    if not raw or raw.lower() == "all":
        return AVAILABLE_MODELS

    aliases = {
        "s-learner_rf": "S-Learner_RF",
        "s_learner_rf": "S-Learner_RF",
        "slearner": "S-Learner_RF",
        "t-learner_rf": "T-Learner_RF",
        "t_learner_rf": "T-Learner_RF",
        "tlearner": "T-Learner_RF",
        "hydranet": "HydraNet",
        "hydra": "HydraNet",
        "medusa": "MEDUSA",
    }
    selected: List[str] = []
    unknown: List[str] = []
    for token in (part.strip() for part in raw.split(",")):
        if not token:
            continue
        canonical = aliases.get(token.lower())
        if canonical is None:
            unknown.append(token)
        elif canonical not in selected:
            selected.append(canonical)
    if unknown:
        raise ValueError(
            f"Unknown model(s): {unknown}. Available models: {list(AVAILABLE_MODELS)}"
        )
    if not selected:
        raise ValueError("At least one model must be selected.")
    return tuple(selected)


def normalize_family_selection(value: str | None) -> Tuple[str, ...]:
    """Return canonical family names from a comma-separated CLI value."""
    raw = "all" if value is None else str(value).strip()
    if not raw or raw.lower() == "all":
        return ("ALL",)

    aliases = {
        "base": "BASE",
        "baseline": "BASE",
        "abc": "ABC",
        "core": "ABC",
        "a": "ABC",
        "b": "ABC",
        "c": "ABC",
        "d": "D",
        "e": "E",
        "f": "F",
        "g": "G",
        "h": "H",
        "i": "I",
        "j": "J",
        "k": "K",
        "k_neighbor": "K_NEIGHBOR",
        "k-neighbor": "K_NEIGHBOR",
        "k_mod5": "K_MOD5",
        "k-mod5": "K_MOD5",
        "k_unsupported": "K_MOD5",
        "k-unsupported": "K_MOD5",
        "k_random": "K_RANDOM",
        "k-random": "K_RANDOM",
        "k_random3": "K_RANDOM3",
        "k-random3": "K_RANDOM3",
        "k_extreme": "K_EXTREME",
        "k-extreme": "K_EXTREME",
        "l": "L",
    }
    selected: List[str] = []
    unknown: List[str] = []
    for token in (part.strip() for part in raw.split(",")):
        if not token:
            continue
        canonical = aliases.get(token.lower())
        if canonical is None:
            unknown.append(token)
        elif canonical not in selected:
            selected.append(canonical)
    if unknown:
        raise ValueError(
            f"Unknown family/families: {unknown}. Available families: "
            f"{list(AVAILABLE_FAMILIES)}"
        )
    if not selected:
        raise ValueError("At least one test family must be selected.")
    return tuple(selected)


def expand_family_dependencies(requested: Sequence[str]) -> Tuple[str, ...]:
    """Compute the minimal transitive family closure required for comparisons.

    Every partial run receives the literal BASE reference.  ``ALL`` remains the
    backward-compatible default and expands to every concrete family.
    """
    requested_set = {str(name).upper() for name in requested}
    if "ALL" in requested_set:
        concrete = set(AVAILABLE_FAMILIES)
        concrete.discard("K")
    else:
        concrete = set(requested_set)
        if "K" in concrete:
            concrete.remove("K")
            concrete.update(K_SUBFAMILIES)
        concrete.add("BASE")

    changed = True
    while changed:
        changed = False
        for family in tuple(concrete):
            for dependency in FAMILY_DEPENDENCIES.get(family, ()):
                if dependency not in concrete:
                    concrete.add(dependency)
                    changed = True

    # Stable display/training order, without the K umbrella pseudo-family.
    order = tuple(name for name in AVAILABLE_FAMILIES if name != "K")
    return tuple(name for name in order if name in concrete)


def scenario_family_selector(scenario: "Scenario") -> str:
    """Map an internal Scenario.family string to its public CLI family name."""
    mapping = {
        "full_2k_given_x": "BASE",
        "core": "ABC",
        "x_support": "D",
        "E_CANONICAL": "E",
        "F_GENERAL_BASIS": "F",
        "G_GENERAL_REDUNDANT": "G",
        "H_LOCAL_SUPPORT_LADDER": "H",
        "I_THEOREM_SAFE_SUPPORT_LADDER": "I",
        "J_RANK4_SUPPORT_SMOOTHING": "J",
        "K_NEIGHBOR_SUPPORT_EXTRAPOLATION": "K_NEIGHBOR",
        "K_UNSUPPORTED_MOD5_EXTRAPOLATION": "K_MOD5",
        "K_RANDOM_MOD8_RANK4_EXTRAPOLATION": "K_RANDOM",
        "K_RANDOM3_MOD8_RANK32_EXTRAPOLATION": "K_RANDOM3",
        "K_EXTREME_MAXMOD_EXTRAPOLATION": "K_EXTREME",
        "L_RANK4_NORM_RICHNESS": "L",
    }
    try:
        return mapping[str(scenario.family)]
    except KeyError as exc:
        raise ValueError(
            f"Scenario {scenario.name!r} has an unregistered family "
            f"{scenario.family!r}."
        ) from exc


def resolve_device(requested: str) -> str:
    """Resolve auto/cpu/cuda once so every neural model uses the same device."""
    torch = getattr(MODELS_MODULE, "torch", None)
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "--device cuda was requested, but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build and verify the NVIDIA driver, "
            "or use --device auto/--device cpu."
        )
    return requested


# =====================================================================
# Treatment geometry
# =====================================================================

def regime_bits_from_codes(codes: np.ndarray, k_bits: int = K_BITS) -> np.ndarray:
    """Little-endian integer code, displayed as (T1,T2,T3,T4)."""
    codes = np.asarray(codes, dtype=int)
    return ((codes[:, None] >> np.arange(k_bits)) & 1).astype(float)


def all_regime_bits(k_bits: int = K_BITS) -> np.ndarray:
    return regime_bits_from_codes(np.arange(2**k_bits), k_bits)


REGIME_BITS = all_regime_bits()
ORIGIN = 0
CANONICAL_NONZERO = (1, 2, 4, 8)  # 1000,0100,0010,0001 in (T1,T2,T3,T4)
CANONICAL_BY_RANK = {
    1: (0, 1),
    2: (0, 1, 2),
    3: (0, 1, 2, 4),
    4: (0, 1, 2, 4, 8),
}


def bit_string(code: int) -> str:
    return "".join(map(str, REGIME_BITS[int(code)].astype(int).tolist()))


def bit_strings(codes: Sequence[int]) -> Tuple[str, ...]:
    return tuple(bit_string(int(c)) for c in codes)


def treatment_matrix_columns(codes: Sequence[int], exclude_origin: bool = True) -> np.ndarray:
    """Formal U: columns are supported treatment vectors u_j in R^K."""
    cc = [int(c) for c in codes if (not exclude_origin or int(c) != ORIGIN)]
    if not cc:
        return np.zeros((K_BITS, 0), dtype=float)
    return REGIME_BITS[np.asarray(cc, dtype=int)].T


def rank_of_codes(codes: Sequence[int]) -> int:
    U = treatment_matrix_columns(codes, exclude_origin=True)
    return int(np.linalg.matrix_rank(U)) if U.size else 0


def geometry_diagnostics(codes: Sequence[int]) -> Dict[str, float]:
    """Rank plus conditioning inside the nonzero span."""
    codes = tuple(sorted(set(map(int, codes))))
    U = treatment_matrix_columns(codes, exclude_origin=True)
    r = int(np.linalg.matrix_rank(U)) if U.size else 0

    if U.size:
        s = np.linalg.svd(U, compute_uv=False)
        s_pos = s[s > 1e-12]
    else:
        s_pos = np.array([], dtype=float)

    sigma_max = float(np.max(s_pos)) if len(s_pos) else 0.0
    sigma_min_pos = float(np.min(s_pos)) if len(s_pos) else 0.0
    cond_pos = float(np.inf if sigma_min_pos <= 1e-12 else sigma_max / sigma_min_pos)

    # Full-space sigma_min is zero whenever rank<K.
    sigma_min_full = sigma_min_pos if r == K_BITS else 0.0

    return {
        "rank_U": r,
        "sigma_min_positive_U": sigma_min_pos,
        "sigma_min_full_U": sigma_min_full,
        "sigma_max_U": sigma_max,
        "condition_nonzero_U": cond_pos,
        "n_support_vertices": len(codes),
        "origin_present": int(ORIGIN in codes),
    }


# =====================================================================
# Base DGP
# =====================================================================

def zscore_population(x: np.ndarray) -> np.ndarray:
    """Exact document form: denominator sqrt(mean squared deviation + 1e-8)."""
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=0, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=0, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS)


def robust_scale_mad(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    return (v - med) / (1.4826 * mad + EPS)


def extrapolation_score(X_star: np.ndarray) -> np.ndarray:
    return (
        1.15 * X_star[:, 0]
        + 0.70 * X_star[:, 2]
        - 0.55 * X_star[:, 8]
        + 0.40 * X_star[:, 6] * X_star[:, 9]
    )


def observation_contamination_probability(X_star: np.ndarray) -> np.ndarray:
    return np.clip(
        OBSERVATION_NOISE
        * (0.35 + 0.65 * HETEROSCEDASTIC_NOISE * np.abs(X_star[:, 0])),
        0.0,
        0.49,
    )


@dataclass
class DGPData:
    seed: int
    X: np.ndarray
    X_struct_raw: np.ndarray
    X_star: np.ndarray
    U_latent: np.ndarray
    A: np.ndarray
    T: np.ndarray
    eta_sys: np.ndarray
    mu_conditional: np.ndarray
    mu_observed_label: np.ndarray
    Ypot_realized: np.ndarray
    Y_factual_clean: np.ndarray
    Yobs: np.ndarray
    factual_prob_realized: np.ndarray
    s_extra: np.ndarray
    assignment_directions: np.ndarray
    interaction_table: pd.DataFrame


def generate_structural_covariates(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.zeros((n, P_STRUCT), dtype=float)

    M = rng.binomial(1, 0.30, size=n)
    X[:, 0] = np.where(
        M == 1,
        rng.normal(-1.2, 0.45, size=n),
        rng.normal(0.8, 1.0, size=n),
    )
    X[:, 1] = rng.standard_t(df=2.5, size=n)
    X[:, 2] = rng.uniform(-2.5, 2.5, size=n)
    B = rng.beta(0.8, 3.5, size=n)
    X[:, 3] = 2.0 * B - 1.0

    D = rng.binomial(1, 0.35, size=n)
    P = rng.poisson(2.5, size=n)
    X[:, 4] = np.where(D == 1, 0.0, P.astype(float))
    X[:, 5] = rng.negative_binomial(3, 0.42, size=n).astype(float)

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    X[:, 6] = np.sin(2.7 * theta + 0.4 * X[:, 3])
    X[:, 7] = np.cos(5.0 * theta) + 0.25 * np.sin(11.0 * theta)

    V = rng.uniform(-2.0, 2.0, size=n)
    E = rng.normal(0.0, 1.0, size=n)
    X[:, 8] = V
    X[:, 9] = -0.7 * V + 0.45 * E + 0.25 * np.sign(V)

    X[:, 10] = rng.exponential(scale=1.0, size=n)
    X[:, 11] = rng.gamma(shape=2.0, scale=0.8, size=n)

    return X, zscore_population(X)


def generate_covariates(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_struct, X_star = generate_structural_covariates(rng, n)
    W = rng.normal(0.0, 1.0, size=(n, P_NOISE))
    return np.column_stack([X_struct, W]), X_struct, X_star


def treatment_assignment(
    rng: np.random.Generator,
    X_star: np.ndarray,
    U_latent: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = X_star.shape[0]

    d0 = rng.normal(0.0, 1.0, size=(P_STRUCT, K_BITS))
    D = d0 / (np.linalg.norm(d0, axis=0, keepdims=True) + EPS)
    q = X_star @ D

    s = OBSERVED_CONFOUNDING_STRENGTH * (
        0.55 * np.sin(1.7 * q)
        + 0.30 * np.tanh(q * X_star[:, [0]])
        + 0.20 * (q > 0.35)
    )

    Sigma = np.full((K_BITS, K_BITS), ERROR_CORRELATION, dtype=float)
    np.fill_diagonal(Sigma, 1.0)
    eps = rng.multivariate_normal(np.zeros(K_BITS), Sigma, size=n)

    ell = np.array([0.65 if j % 2 == 0 else -0.55 for j in range(K_BITS)])
    raw = (
        ASSIGNMENT_STRENGTH * s
        + HIDDEN_CONFOUNDING_STRENGTH * U_latent[:, None] * ell[None, :]
        + eps
    )

    z = np.empty_like(raw)
    for j in range(K_BITS):
        R = rankdata(raw[:, j], method="average")
        z[:, j] = norm.ppf((R - 0.5) / n)

    A = (z < ASSIGNMENT_THRESHOLD).astype(int)
    T = (A * (2 ** np.arange(K_BITS))[None, :]).sum(axis=1).astype(int)
    return A, T, D


def build_heterogeneity_features(X_star: np.ndarray) -> np.ndarray:
    n, p0 = X_star.shape
    h = np.zeros((n, K_BITS), dtype=float)
    for j in range(K_BITS):
        h[:, j] = (
            np.sin((j + 2) * X_star[:, j % p0] + X_star[:, (j + 6) % p0])
            + 0.35
            * np.tanh(
                X_star[:, (j + 4) % p0]
                * X_star[:, (j + 9) % p0]
            )
        )
    return zscore_population(h)


def logistic_normal_mean(
    eta_sys: np.ndarray,
    noise_sd: float,
    tau: float = 1.0,
    n_nodes: int = GH_NODES,
) -> np.ndarray:
    """E_Z expit(clip((eta_sys+Z)/tau,-8,8)), Z~N(0,noise_sd^2)."""
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    shifts = np.sqrt(2.0) * noise_sd * nodes
    out = np.zeros_like(eta_sys, dtype=float)
    for shift, weight in zip(shifts, weights):
        out += weight * expit(np.clip((eta_sys + shift) / tau, -8.0, 8.0))
    return out / np.sqrt(np.pi)


def generate_potential_outcomes(
    rng: np.random.Generator,
    X_star: np.ndarray,
    U_latent: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    n = X_star.shape[0]
    regimes = REGIME_BITS.astype(float)
    p0 = X_star.shape[1]

    b_smooth = (
        0.45 * np.sin(1.4 * X_star[:, 0] * X_star[:, 2])
        - 0.30 * np.cos(2.1 * X_star[:, 6])
        + 0.22 * np.tanh(X_star[:, 1] * X_star[:, 8])
        + 0.18 * (X_star[:, 4] > np.quantile(X_star[:, 4], 0.70))
    )
    b_rugged = (
        0.28 * np.sin(7.0 * X_star[:, 0])
        + 0.25 * np.cos(9.0 * X_star[:, 2] * X_star[:, 6])
        + 0.20 * np.sign(np.sin(5.5 * X_star[:, 8] + X_star[:, 7]))
    )
    eta0 = -0.35 + 0.65 * robust_scale_mad(b_smooth + b_rugged)

    alpha = TREATMENT_SCALE * (0.08 + 0.10 * np.cos(np.arange(K_BITS)))
    eta_main = regimes @ alpha

    h = build_heterogeneity_features(X_star)
    signs = np.array([1.0 if j % 2 == 0 else -1.0 for j in range(K_BITS)])
    individual_h_coeff = h * (HETEROGENEITY_SCALE * 0.24 * signs)[None, :]
    eta_het = individual_h_coeff @ regimes.T

    flat_subsets: List[Tuple[int, ...]] = []
    for order in range(2, MAX_INTERACTION_ORDER + 1):
        flat_subsets.extend(itertools.combinations(range(K_BITS), order))

    eta_int = np.zeros((n, N_REGIMES), dtype=float)
    interaction_terms: List[Tuple[int, Tuple[int, ...], int, float, float]] = []

    for r, subset in enumerate(flat_subsets):
        active = int(rng.binomial(1, INTERACTION_DENSITY))
        sign = -1.0 if r % 3 == 0 else 1.0
        weight = SYNERGY_SCALE / math.sqrt(len(subset))
        g = (
            np.sin((r + 2) * X_star[:, r % p0])
            * np.tanh(X_star[:, (r + 5) % p0])
        )
        I = np.prod(regimes[:, subset], axis=1)
        if active:
            eta_int += sign * weight * g[:, None] * I[None, :]
        interaction_terms.append((r, tuple(subset), active, sign, weight))

    eta_sys = eta0[:, None] + eta_main[None, :] + eta_het + eta_int

    # Base case has hidden confounding strength 0. The branch is retained for fidelity.
    if HIDDEN_CONFOUNDING_STRENGTH > 0:
        L_a = np.sum(
            regimes
            * np.array([0.7 if j % 2 == 0 else -0.6 for j in range(K_BITS)])[None, :],
            axis=1,
        )
        eta_sys += (
            HIDDEN_CONFOUNDING_STRENGTH
            * (U_latent + 0.3 * U_latent**2)[:, None]
            * L_a[None, :]
        )

    xi_i = rng.normal(0.0, LATENT_NOISE_SD, size=n)
    xi_ia = rng.normal(0.0, REGIME_LATENT_NOISE_SD, size=(n, N_REGIMES))
    eta_realized = eta_sys + xi_i[:, None] + xi_ia
    tau = max(0.05, OUTCOME_TEMPERATURE)
    Ypot_realized = expit(np.clip(eta_realized / tau, -8.0, 8.0))

    total_noise_sd = math.sqrt(LATENT_NOISE_SD**2 + REGIME_LATENT_NOISE_SD**2)
    mu_conditional = logistic_normal_mean(eta_sys, total_noise_sd, tau=tau)

    interaction_df = pd.DataFrame(
        interaction_terms,
        columns=["r", "subset", "active", "sign", "weight"],
    )
    return eta_sys, mu_conditional, Ypot_realized, interaction_df


def generate_observed_outcome(
    rng: np.random.Generator,
    Ypot_realized: np.ndarray,
    mu_conditional: np.ndarray,
    T: np.ndarray,
    X_star: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(T)
    factual_prob_realized = Ypot_realized[np.arange(n), T]
    y_clean = rng.binomial(1, factual_prob_realized, size=n).astype(float)

    c = observation_contamination_probability(X_star)
    contaminated = rng.binomial(1, c, size=n).astype(bool)
    fair_coin = rng.binomial(1, 0.5, size=n).astype(float)
    y_obs = np.where(contaminated, fair_coin, y_clean)

    mu_observed_label = (1.0 - c[:, None]) * mu_conditional + 0.5 * c[:, None]
    return y_clean, y_obs, factual_prob_realized, mu_observed_label


def generate_base_dgp(seed: int, n: int = N) -> DGPData:
    if n < 2:
        raise ValueError(f"n must be at least 2; received {n}.")

    rng = np.random.default_rng(seed)
    X, X_struct_raw, X_star = generate_covariates(rng, n)
    U_latent = rng.normal(0.0, 1.0, size=n)
    A, T, directions = treatment_assignment(rng, X_star, U_latent)
    eta_sys, mu_conditional, Ypot_realized, interaction_table = generate_potential_outcomes(
        rng, X_star, U_latent
    )
    y_clean, y_obs, factual_prob_realized, mu_observed_label = generate_observed_outcome(
        rng, Ypot_realized, mu_conditional, T, X_star
    )

    return DGPData(
        seed=seed,
        X=X,
        X_struct_raw=X_struct_raw,
        X_star=X_star,
        U_latent=U_latent,
        A=A,
        T=T,
        eta_sys=eta_sys,
        mu_conditional=mu_conditional,
        mu_observed_label=mu_observed_label,
        Ypot_realized=Ypot_realized,
        Y_factual_clean=y_clean,
        Yobs=y_obs,
        factual_prob_realized=factual_prob_realized,
        s_extra=extrapolation_score(X_star),
        assignment_directions=directions,
        interaction_table=interaction_table,
    )


# =====================================================================
# X strata and common population evaluation set
# =====================================================================

def define_x_regions(data: DGPData) -> Tuple[np.ndarray, Dict[str, float]]:
    q25, q50, q75 = np.quantile(data.s_extra, [0.25, 0.50, 0.75])
    b = np.digitize(data.s_extra, [q25, q50, q75], right=True)
    labels = np.asarray([X_REGIONS[i] for i in b], dtype=object)
    return labels, {"q25": float(q25), "q50": float(q50), "q75": float(q75)}


def common_population_eval_split(
    region_labels: np.ndarray,
    seed: int,
    eval_fraction: float = POP_EVAL_FRACTION,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reserve the same representative X sample for all scenario-level causal metrics."""
    rng = np.random.default_rng(seed)
    eval_parts, train_parts = [], []
    for region in X_REGIONS:
        ids = np.where(region_labels == region)[0].copy()
        rng.shuffle(ids)
        n_eval = int(round(eval_fraction * len(ids)))
        n_eval = min(max(n_eval, 1), len(ids) - 1)
        eval_parts.append(ids[:n_eval])
        train_parts.append(ids[n_eval:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(eval_parts))


def cyclic_rank_map(rotation: int) -> Dict[str, int]:
    """Latin/cyclic counterbalancing of physical X region and local treatment rank."""
    ranks = [1, 2, 3, 4]
    shift = int(rotation) % 4
    ranks = ranks[shift:] + ranks[:shift]
    return dict(zip(X_REGIONS, ranks))


# =====================================================================
# Scenario representation and sampling
# =====================================================================

@dataclass
class Scenario:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    description: str
    local_supports: Dict[str, Tuple[int, ...]]
    local_bases: Dict[str, Tuple[int, ...]]
    target_kind: str = "population"
    family: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    # Optional synthetic/expanded training design. When supplied, each row uses
    # X[train_source_idx_override] with treatment train_t_override instead of
    # the factual T attached to the original unit. This is used by the literal
    # full 2^K|x baseline.
    train_source_idx_override: np.ndarray | None = None
    train_t_override: np.ndarray | None = None
    train_y_clean_override: np.ndarray | None = None
    train_y_contaminated_override: np.ndarray | None = None

    def has_training_override(self) -> bool:
        return self.train_source_idx_override is not None and self.train_t_override is not None

    def training_source_idx(self) -> np.ndarray:
        return (
            np.asarray(self.train_source_idx_override, dtype=int)
            if self.has_training_override()
            else np.asarray(self.train_idx, dtype=int)
        )

    def training_t(self, data: "DGPData") -> np.ndarray:
        return (
            np.asarray(self.train_t_override, dtype=int)
            if self.has_training_override()
            else np.asarray(data.T[self.train_idx], dtype=int)
        )

    def n_training_rows(self) -> int:
        return int(len(self.train_t_override)) if self.has_training_override() else int(len(self.train_idx))

    def n_unique_training_x(self) -> int:
        return int(len(np.unique(self.training_source_idx())))

    def global_observed_support(self, data: "DGPData") -> Tuple[int, ...]:
        return tuple(sorted(np.unique(self.training_t(data)).astype(int).tolist()))


def candidate_pool(
    data: DGPData,
    train_pool: np.ndarray,
    region_labels: np.ndarray,
    region: str,
    support: Sequence[int],
) -> np.ndarray:
    ids = train_pool[region_labels[train_pool] == region]
    return ids[np.isin(data.T[ids], np.asarray(support, dtype=int))]


def sample_with_minimum_support(
    candidate_idx: np.ndarray,
    T: np.ndarray,
    support: Sequence[int],
    basis: Sequence[int],
    n_target: int,
    rng: np.random.Generator,
    basis_min_share: float = MIN_BASIS_EMPIRICAL_SHARE,
    require_all_support: bool = True,
) -> np.ndarray:
    """
    Sample without replacement, guaranteeing baseline and basis empirical mass.

    For strong positivity A, require_all_support=True reserves at least one of
    every supported vertex. For weak/full-rank conditions, baseline and each
    designated spanning vector receive at least ceil(min_share*n_target).
    """
    candidate_idx = np.asarray(candidate_idx, dtype=int)
    support = tuple(sorted(set(map(int, support))))
    basis = tuple(sorted(set(map(int, basis))))

    if n_target > len(candidate_idx):
        raise RuntimeError("Requested local sample exceeds candidate pool.")

    minimums: Dict[int, int] = {}
    if require_all_support:
        minimums.update({t: 1 for t in support})

    protected = set(basis) | {ORIGIN}
    min_protected = max(1, int(math.ceil(basis_min_share * n_target)))
    for t in protected:
        if t in support:
            minimums[t] = max(minimums.get(t, 0), min_protected)

    if sum(minimums.values()) > n_target:
        raise RuntimeError("Minimum support reservations exceed local n_target.")

    selected: List[int] = []
    selected_set: set[int] = set()

    for t, m in sorted(minimums.items()):
        ids = candidate_idx[T[candidate_idx] == t]
        if len(ids) < m:
            raise RuntimeError(
                f"Treatment {bit_string(t)} has {len(ids)} candidates but {m} are required."
            )
        chosen = rng.choice(ids, size=m, replace=False)
        selected.extend(map(int, chosen))
        selected_set.update(map(int, chosen))

    remaining_pool = np.asarray(
        [i for i in candidate_idx if int(i) not in selected_set], dtype=int
    )
    n_more = n_target - len(selected)
    if n_more > len(remaining_pool):
        raise RuntimeError("Insufficient remaining candidate observations.")
    if n_more:
        selected.extend(map(int, rng.choice(remaining_pool, size=n_more, replace=False)))

    out = np.asarray(selected, dtype=int)
    rng.shuffle(out)
    return out


def local_capacity(
    data: DGPData,
    train_pool: np.ndarray,
    region_labels: np.ndarray,
    region: str,
    support: Sequence[int],
) -> int:
    return len(candidate_pool(data, train_pool, region_labels, region, support))


def equal_region_total_cap(
    data: DGPData,
    train_pool: np.ndarray,
    region_labels: np.ndarray,
    supports_list: Sequence[Mapping[str, Sequence[int]]],
    max_train_n: int | None,
) -> int:
    """Common total n for scenario sets with equal 25% X-region shares."""
    local_cap = np.inf
    for supports in supports_list:
        for region in X_REGIONS:
            local_cap = min(
                local_cap,
                local_capacity(data, train_pool, region_labels, region, supports[region]),
            )
    per_region = int(local_cap)
    if max_train_n is not None:
        per_region = min(per_region, int(max_train_n) // 4)
    if per_region < 64:
        raise RuntimeError(f"Too few observations per X region: {per_region}")
    return 4 * per_region


def build_equal_region_scenario(
    data: DGPData,
    train_pool: np.ndarray,
    test_idx: np.ndarray,
    region_labels: np.ndarray,
    name: str,
    local_supports: Mapping[str, Sequence[int]],
    local_bases: Mapping[str, Sequence[int]],
    total_n: int,
    seed: int,
    description: str,
    family: str,
) -> Scenario:
    if total_n % 4 != 0:
        total_n -= total_n % 4
    n_r = total_n // 4
    pieces = []
    for j, region in enumerate(X_REGIONS):
        pool = candidate_pool(data, train_pool, region_labels, region, local_supports[region])
        ids = sample_with_minimum_support(
            pool,
            data.T,
            local_supports[region],
            local_bases[region],
            n_r,
            np.random.default_rng(seed + 101 * j),
            require_all_support=True,
        )
        pieces.append(ids)
    tr = np.concatenate(pieces)
    np.random.default_rng(seed + 999).shuffle(tr)
    return Scenario(
        name=name,
        train_idx=tr,
        test_idx=np.asarray(test_idx, dtype=int),
        description=description,
        local_supports={k: tuple(map(int, v)) for k, v in local_supports.items()},
        local_bases={k: tuple(map(int, v)) for k, v in local_bases.items()},
        family=family,
    )



def build_full_2k_given_x_baseline(
    data: DGPData,
    train_pool: np.ndarray,
    population_eval_idx: np.ndarray,
    region_labels: np.ndarray,
    seed: int,
    max_train_n: int | None,
) -> Scenario:
    """
    Literal 2^K | x reference.

    A stratified subset of unique training X values is repeated once for each
    treatment t in {0,...,15}.  The expanded number of rows never exceeds the
    available training-unit budget (nor max_train_n, when supplied). Hence the
    empirical treatment support at every selected training x is exactly all 16
    vertices without multiplying the requested total sample size by 16.

    Clean labels are generated exactly as the source DGP would generate a
    Bernoulli outcome from the realization-specific potential probability
    Ypot_realized(x,t). Contaminated labels then apply the same heteroscedastic
    replacement-by-fair-coin mechanism used by the source benchmark.
    """
    row_budget = len(train_pool)
    if max_train_n is not None:
        row_budget = min(row_budget, int(max_train_n))
    n_anchors = row_budget // N_REGIMES
    if n_anchors < len(X_REGIONS):
        raise RuntimeError(
            "The full 2^K|x baseline needs enough training budget for at least "
            f"one covariate anchor per X region ({len(X_REGIONS) * N_REGIMES} rows)."
        )

    # Keep the four physical X strata represented while respecting the total
    # row budget.  Each selected anchor contributes exactly 16 training rows.
    rng = np.random.default_rng(seed)
    anchors_per_region = n_anchors // len(X_REGIONS)
    remainder = n_anchors % len(X_REGIONS)
    anchor_parts: List[np.ndarray] = []
    for region_pos, region in enumerate(X_REGIONS):
        candidates = np.asarray(
            train_pool[region_labels[train_pool] == region], dtype=int
        )
        n_region = anchors_per_region + int(region_pos < remainder)
        if len(candidates) < n_region:
            raise RuntimeError(
                f"Insufficient {region} anchors for the full 2^K|x baseline: "
                f"need {n_region}, found {len(candidates)}."
            )
        anchor_parts.append(rng.choice(candidates, size=n_region, replace=False))

    x_idx = np.concatenate(anchor_parts)
    rng.shuffle(x_idx)
    source_idx = np.repeat(x_idx, N_REGIMES)
    t_override = np.tile(np.arange(N_REGIMES, dtype=int), len(x_idx))

    p_realized = data.Ypot_realized[source_idx, t_override]
    y_clean = rng.binomial(1, p_realized).astype(float)

    c = observation_contamination_probability(data.X_star[source_idx])
    replace = rng.binomial(1, c).astype(bool)
    fair_coin = rng.binomial(1, 0.5, size=len(source_idx)).astype(float)
    y_contaminated = np.where(replace, fair_coin, y_clean)

    support = tuple(range(N_REGIMES))
    supports = {r: support for r in X_REGIONS}
    bases = {r: CANONICAL_NONZERO for r in X_REGIONS}

    return Scenario(
        name=FULL2K_BASELINE_NAME,
        train_idx=x_idx,  # unique X anchors; expanded rows live in overrides
        test_idx=np.asarray(population_eval_idx, dtype=int),
        description=(
            "Literal fully crossed reference: every training x is paired with all "
            "2^K=16 treatment actions generated from the same potential-outcome DGP."
        ),
        local_supports=supports,
        local_bases=bases,
        target_kind="population",
        family="full_2k_given_x",
        metadata={
            "literal_all_2k_per_x": 1,
            "training_row_budget": int(row_budget),
            "n_unique_train_x": int(len(x_idx)),
            "n_expanded_train_rows": int(len(source_idx)),
        },
        train_source_idx_override=source_idx,
        train_t_override=t_override,
        train_y_clean_override=y_clean,
        train_y_contaminated_override=y_contaminated,
    )


def build_abc_scenarios(
    data: DGPData,
    train_pool: np.ndarray,
    population_eval_idx: np.ndarray,
    region_labels: np.ndarray,
    seed: int,
    max_train_n: int | None,
) -> Tuple[Dict[str, Scenario], Dict[str, object]]:
    """
    Core theorem-facing comparison.

    A_STRONG: all 16 vertices locally present.
    B_WEAK: baseline + canonical full-rank generating set locally present.
    C_RANK3: baseline + rank-3 canonical generating set locally present.
    """
    support_A = tuple(range(N_REGIMES))
    support_B = CANONICAL_BY_RANK[4]
    support_C = CANONICAL_BY_RANK[3]

    supports_A = {r: support_A for r in X_REGIONS}
    supports_B = {r: support_B for r in X_REGIONS}
    supports_C = {r: support_C for r in X_REGIONS}

    bases_A = {r: CANONICAL_NONZERO for r in X_REGIONS}
    bases_B = {r: CANONICAL_NONZERO for r in X_REGIONS}
    bases_C = {r: CANONICAL_NONZERO[:3] for r in X_REGIONS}

    common_n = equal_region_total_cap(
        data,
        train_pool,
        region_labels,
        [supports_A, supports_B, supports_C],
        max_train_n,
    )

    scenarios = {
        "A_STRONG": build_equal_region_scenario(
            data, train_pool, population_eval_idx, region_labels,
            "A_STRONG", supports_A, bases_A, common_n, seed + 1000,
            "All 16 treatment vertices represented in every X stratum.",
            "core",
        ),
        "B_WEAK": build_equal_region_scenario(
            data, train_pool, population_eval_idx, region_labels,
            "B_WEAK", supports_B, bases_B, common_n, seed + 2000,
            "Baseline plus canonical full-rank treatment generating set in every X stratum.",
            "core",
        ),
        "C_RANK3": build_equal_region_scenario(
            data, train_pool, population_eval_idx, region_labels,
            "C_RANK3", supports_C, bases_C, common_n, seed + 3000,
            "Baseline remains supported, but local treatment support has rank 3<K.",
            "core",
        ),
    }
    return scenarios, {"common_train_n": common_n}


def build_d_scenarios(
    data: DGPData,
    train_pool: np.ndarray,
    population_eval_idx: np.ndarray,
    region_labels: np.ndarray,
    seed: int,
    max_train_n: int | None,
) -> Tuple[Dict[str, Scenario], Dict[str, object]]:
    """Full treatment rank plus good vs absent X support in Q4."""
    support = CANONICAL_BY_RANK[4]
    basis = CANONICAL_NONZERO
    supports_ref = {r: support for r in X_REGIONS}
    bases_ref = {r: basis for r in X_REGIONS}

    tail_eval = population_eval_idx[region_labels[population_eval_idx] == "X_Q4"]
    if len(tail_eval) < 128:
        raise RuntimeError("Tail evaluation set too small.")

    # REF: equal coverage over all four X regions.
    cap_ref = min(
        local_capacity(data, train_pool, region_labels, r, support) for r in X_REGIONS
    )
    n_ref = 4 * int(cap_ref)

    # POOR: no Q4 training, equal mixture over Q1..Q3.
    cap_poor = min(
        local_capacity(data, train_pool, region_labels, r, support) for r in X_REGIONS[:3]
    )
    n_poor = 3 * int(cap_poor)

    total_n = min(n_ref, n_poor)
    if max_train_n is not None:
        total_n = min(total_n, int(max_train_n))
    total_n = (total_n // 12) * 12  # divisible by 4 and 3
    if total_n < 384:
        raise RuntimeError("D scenarios have insufficient training data.")

    ref = build_equal_region_scenario(
        data, train_pool, tail_eval, region_labels,
        "D_REF_X", supports_ref, bases_ref, total_n, seed + 4000,
        "Canonical full-rank support with X coverage including the target Q4 region.",
        "x_support",
    )

    # D_POOR_X: construct manually with no Q4 training.
    pieces = []
    n_r = total_n // 3
    for j, region in enumerate(X_REGIONS[:3]):
        pool = candidate_pool(data, train_pool, region_labels, region, support)
        ids = sample_with_minimum_support(
            pool, data.T, support, basis, n_r,
            np.random.default_rng(seed + 5000 + 101*j),
            require_all_support=True,
        )
        pieces.append(ids)
    tr_poor = np.concatenate(pieces)
    np.random.default_rng(seed + 5999).shuffle(tr_poor)
    supports_poor = {r: support for r in X_REGIONS[:3]}
    supports_poor["X_Q4"] = tuple()
    bases_poor = {r: basis for r in X_REGIONS[:3]}
    bases_poor["X_Q4"] = tuple()

    poor = Scenario(
        name="D_POOR_X",
        train_idx=tr_poor,
        test_idx=tail_eval,
        description=(
            "Canonical full-rank treatment support in observed X strata, but the Q4 target "
            "region is absent from training."
        ),
        local_supports=supports_poor,
        local_bases=bases_poor,
        target_kind="x_tail",
        family="x_support",
    )
    ref.target_kind = "x_tail"

    return {"D_REF_X": ref, "D_POOR_X": poor}, {"common_train_n": total_n}


# =====================================================================
# General support generation for regional tests
# =====================================================================

def best_basis_inside(codes: Sequence[int], target_rank: int) -> Tuple[int, ...] | None:
    nz = [int(c) for c in codes if int(c) != ORIGIN]
    best = None
    best_sigma = -np.inf
    for basis in itertools.combinations(nz, target_rank):
        if rank_of_codes(basis) != target_rank:
            continue
        s = np.linalg.svd(treatment_matrix_columns(basis), compute_uv=False)
        sigma = float(np.min(s)) if len(s) else 0.0
        if sigma > best_sigma:
            best_sigma = sigma
            best = tuple(map(int, basis))
    return best


def choose_general_support(
    data: DGPData,
    train_pool: np.ndarray,
    region_labels: np.ndarray,
    region: str,
    target_rank: int,
    n_nonzero: int,
    rng: np.random.Generator,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Choose an arbitrary binary support with exact rank and baseline 0000."""
    ids = train_pool[region_labels[train_pool] == region]
    counts = np.bincount(data.T[ids], minlength=N_REGIMES)
    available = [t for t in range(1, N_REGIMES) if counts[t] > 0]

    if n_nonzero < target_rank:
        raise ValueError("n_nonzero must be >= target_rank")

    canonical_set = set(CANONICAL_NONZERO[:target_rank])
    candidates = []

    for combo in itertools.combinations(available, n_nonzero):
        if rank_of_codes(combo) != target_rank:
            continue
        basis = best_basis_inside(combo, target_rank)
        if basis is None:
            continue
        min_count = min([counts[0]] + [counts[t] for t in combo])
        total_count = int(counts[0] + sum(counts[t] for t in combo))
        s = np.linalg.svd(treatment_matrix_columns(basis), compute_uv=False)
        sigma = float(np.min(s)) if len(s) else 0.0
        noncanonical = int(set(combo) != canonical_set)
        score = (noncanonical, min_count, total_count, sigma)
        candidates.append((score, tuple(combo), tuple(basis)))

    if not candidates:
        raise RuntimeError(
            f"{region}: no arbitrary support with n_nonzero={n_nonzero}, rank={target_rank}."
        )

    candidates.sort(key=lambda z: z[0], reverse=True)
    best_noncanonical = candidates[0][0][0]
    same_kind = [c for c in candidates if c[0][0] == best_noncanonical]
    best_min_count = max(c[0][1] for c in same_kind)
    # Randomize orientation only among supports with substantial empirical mass;
    # the formal assumption requires a positive lower bound epsilon_U, so a
    # basis carried by only a handful of observations is not an appropriate
    # controlled realization of the assumption.
    eligible = [c for c in same_kind if c[0][1] >= 0.75 * best_min_count]
    eligible.sort(key=lambda z: z[0], reverse=True)
    top = eligible[: min(12, len(eligible))]
    _, combo, basis = top[int(rng.integers(0, len(top)))]
    support = tuple(sorted((ORIGIN, *combo)))
    if rank_of_codes(support) != target_rank:
        raise AssertionError("General support rank construction failed.")
    return support, basis


def supports_for_rank_map_canonical(rank_map: Mapping[str, int]) -> Tuple[Dict[str, Tuple[int, ...]], Dict[str, Tuple[int, ...]]]:
    supports, bases = {}, {}
    for region, r in rank_map.items():
        supports[region] = CANONICAL_BY_RANK[int(r)]
        bases[region] = CANONICAL_NONZERO[: int(r)]
    return supports, bases


def supports_for_rank_map_general(
    data: DGPData,
    train_pool: np.ndarray,
    region_labels: np.ndarray,
    rank_map: Mapping[str, int],
    seed: int,
    redundant: bool,
) -> Tuple[Dict[str, Tuple[int, ...]], Dict[str, Tuple[int, ...]]]:
    rng = np.random.default_rng(seed)
    supports, bases = {}, {}
    redundant_sizes = {1: 1, 2: 3, 3: 4, 4: 5}
    for region in X_REGIONS:
        r = int(rank_map[region])
        n_nonzero = redundant_sizes[r] if redundant else r
        support, basis = choose_general_support(
            data, train_pool, region_labels, region, r, n_nonzero, rng
        )
        supports[region] = support
        bases[region] = basis
    return supports, bases


def mixture_counts(total_n: int, dominant_region: str, dominant_share: float) -> Dict[str, int]:
    if dominant_region not in X_REGIONS:
        raise ValueError(dominant_region)
    if not (0.25 < dominant_share < 1.0):
        raise ValueError("dominant_share must be in (0.25,1)")
    n_dom = int(round(total_n * dominant_share))
    n_dom = min(max(n_dom, 1), total_n - 3)
    rem = total_n - n_dom
    others = [r for r in X_REGIONS if r != dominant_region]
    base, extra = divmod(rem, 3)
    out = {dominant_region: n_dom}
    for j, region in enumerate(others):
        out[region] = base + int(j < extra)
    return out


def regional_family_cap(
    data: DGPData,
    train_pool: np.ndarray,
    region_labels: np.ndarray,
    supports: Mapping[str, Sequence[int]],
    bases: Mapping[str, Sequence[int]],
    rank_map: Mapping[str, int],
    dominant_share: float,
    max_train_n: int | None,
) -> int:
    """Largest n feasible under the actual local treatment counts and epsilon guards."""
    cap = int(max_train_n or len(train_pool))
    cap = (cap // 20) * 20
    for total in range(cap, 399, -20):
        feasible = True
        for dominant_rank in (1, 2, 3, 4):
            dominant_region = next(r for r in X_REGIONS if rank_map[r] == dominant_rank)
            alloc = mixture_counts(total, dominant_region, dominant_share)
            for region, n_target in alloc.items():
                pool = candidate_pool(data, train_pool, region_labels, region, supports[region])
                if n_target > len(pool):
                    feasible = False
                    break
                counts = np.bincount(data.T[pool], minlength=N_REGIMES)
                min_protected = max(1, int(math.ceil(MIN_BASIS_EMPIRICAL_SHARE * n_target)))
                protected = set(map(int, bases[region])) | {ORIGIN}
                if any(counts[t] < min_protected for t in protected):
                    feasible = False
                    break
                if any(counts[t] < 1 for t in supports[region]):
                    feasible = False
                    break
                minimum_total = sum(
                    max(1, min_protected if t in protected else 1)
                    for t in set(map(int, supports[region]))
                )
                if minimum_total > n_target:
                    feasible = False
                    break
            if not feasible:
                break
        if feasible:
            return total
    raise RuntimeError("Could not find feasible regional family n_train >=400.")


def build_regional_family(
    data: DGPData,
    train_pool: np.ndarray,
    population_eval_idx: np.ndarray,
    region_labels: np.ndarray,
    rank_map: Mapping[str, int],
    supports: Mapping[str, Sequence[int]],
    bases: Mapping[str, Sequence[int]],
    prefix: str,
    description_family: str,
    total_n: int,
    dominant_share: float,
    seed: int,
) -> Dict[str, Scenario]:
    out = {}
    for s_ix, dominant_rank in enumerate((4, 3, 2, 1)):
        dominant_region = next(r for r in X_REGIONS if int(rank_map[r]) == dominant_rank)
        counts = mixture_counts(total_n, dominant_region, dominant_share)
        pieces = []
        for r_ix, region in enumerate(X_REGIONS):
            pool = candidate_pool(data, train_pool, region_labels, region, supports[region])
            ids = sample_with_minimum_support(
                pool,
                data.T,
                supports[region],
                bases[region],
                counts[region],
                np.random.default_rng(seed + 1000*s_ix + 101*r_ix),
                require_all_support=True,
            )
            pieces.append(ids)
        tr = np.concatenate(pieces)
        np.random.default_rng(seed + 9000 + s_ix).shuffle(tr)
        name = f"{prefix}_RANK{dominant_rank}_DOM"
        out[name] = Scenario(
            name=name,
            train_idx=tr,
            test_idx=population_eval_idx,
            description=(
                f"{description_family}; local ranks 1..4 across X strata, with the rank-"
                f"{dominant_rank} stratum contributing {dominant_share:.0%} of training data."
            ),
            local_supports={k: tuple(map(int, v)) for k, v in supports.items()},
            local_bases={k: tuple(map(int, v)) for k, v in bases.items()},
            family=prefix,
            metadata={"dominant_rank": dominant_rank, "dominant_region": dominant_region},
        )
    return out


# ---------------------------------------------------------------------
# L: rank-4 size-5 supports with progressively richer treatment norms
# ---------------------------------------------------------------------

L_PROFILE_ORDER = (
    "NORM1",
    "NORM12",
    "NORM123",
    "NORM1234",
)

# Here "norm" means the Hamming/L0 norm ||t||_0, i.e. the number of active
# treatment components in a binary treatment vector.  Each profile has exactly
# four nonzero vectors plus the origin, and the four nonzero vectors must be
# linearly independent.  The repeated weight in NORM12/NORM123 keeps a smooth
# progression in mean treatment order while preserving size 5 throughout.
L_WEIGHT_PATTERNS: Dict[str, Tuple[int, int, int, int]] = {
    "NORM1": (1, 1, 1, 1),
    "NORM12": (1, 1, 2, 2),
    "NORM123": (1, 2, 2, 3),
    "NORM1234": (1, 2, 3, 4),
}

L_SCENARIO_NAMES = {
    "NORM1": "L_RANK4_NORM1_DOM",
    "NORM12": "L_RANK4_NORM12_DOM",
    "NORM123": "L_RANK4_NORM123_DOM",
    "NORM1234": "L_RANK4_NORM1234_DOM",
}


def treatment_hamming_weight(code: int) -> int:
    """Number of active treatment bits ||t||_0 for a regime code."""
    return int(np.sum(REGIME_BITS[int(code)]))


def cyclic_l_profile_map(rotation: int) -> Dict[str, str]:
    """Counterbalance L support profiles over physical X regions across seeds."""
    profiles = list(L_PROFILE_ORDER)
    shift = int(rotation) % len(profiles)
    profiles = profiles[shift:] + profiles[:shift]
    return dict(zip(X_REGIONS, profiles))


def enumerate_rank4_bases_by_weight_pattern(
    weight_pattern: Sequence[int],
) -> Tuple[Tuple[int, ...], ...]:
    """All four-vector binary bases with the requested sorted Hamming weights."""
    target = tuple(sorted(map(int, weight_pattern)))
    out: List[Tuple[int, ...]] = []
    for comb in itertools.combinations(range(1, N_REGIMES), K_BITS):
        if tuple(sorted(treatment_hamming_weight(c) for c in comb)) != target:
            continue
        if rank_of_codes(comb) != K_BITS:
            continue
        out.append(tuple(sorted(map(int, comb))))
    if not out:
        raise RuntimeError(f"No rank-4 basis exists for L weight pattern {target}.")
    return tuple(out)


L_BASIS_CATALOGS: Dict[str, Tuple[Tuple[int, ...], ...]] = {
    profile: enumerate_rank4_bases_by_weight_pattern(pattern)
    for profile, pattern in L_WEIGHT_PATTERNS.items()
}


def choose_l_rank4_basis(
    profile: str,
    rng: np.random.Generator,
) -> Tuple[int, ...]:
    """Choose an L basis from the exact Hamming-weight catalog.

    L is intended to isolate treatment-basis composition rather than empirical
    assignment propensity.  Hence noncanonical orientations are sampled from the
    full combinatorial catalog without looking at T frequencies or outcomes.
    Across seeds this counterbalances which specific joint-treatment directions
    instantiate a given Hamming-weight profile.
    """
    if profile not in L_BASIS_CATALOGS:
        raise ValueError(f"Unknown L profile: {profile}")
    catalog = L_BASIS_CATALOGS[profile]
    chosen = catalog[int(rng.integers(0, len(catalog)))]
    if rank_of_codes(chosen) != K_BITS:
        raise AssertionError("L basis selector returned a non-rank-4 basis.")
    observed_pattern = tuple(sorted(treatment_hamming_weight(c) for c in chosen))
    if observed_pattern != L_WEIGHT_PATTERNS[profile]:
        raise AssertionError("L basis selector returned the wrong Hamming-weight pattern.")
    return tuple(map(int, chosen))


def l_anchor_family_cap(
    train_pool: np.ndarray,
    region_labels: np.ndarray,
    dominant_share: float,
    max_train_n: int | None,
) -> int:
    """Largest common number of X anchors for the four L dominance scenarios.

    Each selected anchor is crossed with exactly five local treatment vertices,
    so the expanded L training design respects the same total row budget used by
    the other controlled scenarios.  Feasibility depends only on available X
    anchors, not on factual treatment frequencies.
    """
    row_budget = len(train_pool)
    if max_train_n is not None:
        row_budget = min(row_budget, int(max_train_n))
    max_anchors = row_budget // (K_BITS + 1)
    max_anchors = max_anchors - (max_anchors % 4)
    region_capacity = {
        region: int(np.sum(region_labels[train_pool] == region)) for region in X_REGIONS
    }
    for total_anchors in range(max_anchors, 15, -4):
        feasible = True
        for dominant_region in X_REGIONS:
            alloc = mixture_counts(total_anchors, dominant_region, dominant_share)
            if any(alloc[r] > region_capacity[r] for r in X_REGIONS):
                feasible = False
                break
        if feasible:
            return int(total_anchors)
    raise RuntimeError("Could not find a feasible L crossed-support anchor budget.")


def _crossed_l_region_block(
    data: DGPData,
    anchors: np.ndarray,
    support: Sequence[int],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Create controlled training rows for one L region and one fixed support.

    Every selected X anchor contributes one row for each of the five supported
    treatments.  Labels are generated from the same DGP potential-outcome and
    contamination mechanisms used by the literal full-2^K|x baseline.
    """
    anchors = np.asarray(anchors, dtype=int)
    support_arr = np.asarray(tuple(map(int, support)), dtype=int)
    source_idx = np.repeat(anchors, len(support_arr))
    t_override = np.tile(support_arr, len(anchors))

    p_realized = data.Ypot_realized[source_idx, t_override]
    y_clean = rng.binomial(1, p_realized).astype(float)
    c = observation_contamination_probability(data.X_star[source_idx])
    replace = rng.binomial(1, c).astype(bool)
    fair_coin = rng.binomial(1, 0.5, size=len(source_idx)).astype(float)
    y_contaminated = np.where(replace, fair_coin, y_clean)
    return {
        "anchors": anchors,
        "source_idx": source_idx,
        "t": t_override,
        "y_clean": y_clean,
        "y_contaminated": y_contaminated,
    }


def build_l_rank4_norm_richness_scenarios(
    data: DGPData,
    train_pool: np.ndarray,
    population_eval_idx: np.ndarray,
    region_labels: np.ndarray,
    rotation: int,
    seed: int,
    dominant_share: float,
    max_train_n: int | None,
) -> Tuple[Dict[str, Scenario], Dict[str, object]]:
    """Build benchmark L: rank 4/size 5 everywhere, varying treatment order.

    L is a deliberately controlled support-design experiment.  Four Hamming-
    weight profiles are assigned one-to-one to the four physical X regions and
    rotated across repetitions.  Within a region, selected X anchors are crossed
    with the five local support vertices (origin + four independent treatments).
    This removes factual-propensity scarcity as a confound: in particular, the
    unique norm-4 treatment 1111 receives the same number of training rows as
    each other supported vertex whenever it belongs to a local basis.

    The X-selection strategy mirrors H/regional tests: four scenarios use the
    same profile-to-region map and make one region contribute ``dominant_share``
    of the training anchors, with the other three sharing the remainder.  Master
    anchor lists and generated labels are shared, so rows retained in multiple L
    scenarios are literally identical.  Evaluation always asks for all 16
    potential outcomes on the common population evaluation set.
    """
    profile_map = cyclic_l_profile_map(rotation)
    supports: Dict[str, Tuple[int, ...]] = {}
    bases: Dict[str, Tuple[int, ...]] = {}

    for r_ix, region in enumerate(X_REGIONS):
        profile = str(profile_map[region])
        basis = choose_l_rank4_basis(
            profile,
            np.random.default_rng(seed + 101 * r_ix),
        )
        support = tuple(sorted((ORIGIN, *basis)))
        if len(support) != K_BITS + 1 or rank_of_codes(support) != K_BITS:
            raise AssertionError(
                f"{region}: L profile {profile} must have rank 4 and size 5."
            )
        supports[region] = support
        bases[region] = basis

    total_anchors = l_anchor_family_cap(
        train_pool,
        region_labels,
        dominant_share,
        max_train_n,
    )

    # Determine the largest per-region anchor requirement over the four matched
    # dominance scenarios, then draw one master list per physical X region.
    allocations: Dict[str, Dict[str, int]] = {}
    max_needed = {region: 0 for region in X_REGIONS}
    for dominant_profile in L_PROFILE_ORDER:
        dominant_region = next(
            r for r in X_REGIONS if str(profile_map[r]) == dominant_profile
        )
        alloc = mixture_counts(total_anchors, dominant_region, dominant_share)
        allocations[dominant_profile] = alloc
        for region in X_REGIONS:
            max_needed[region] = max(max_needed[region], int(alloc[region]))

    master_blocks: Dict[str, Dict[str, np.ndarray]] = {}
    for r_ix, region in enumerate(X_REGIONS):
        candidates = np.asarray(
            train_pool[region_labels[train_pool] == region], dtype=int
        )
        need = int(max_needed[region])
        if len(candidates) < need:
            raise RuntimeError(
                f"{region}: L needs {need} X anchors but only {len(candidates)} are available."
            )
        anchor_rng = np.random.default_rng(seed + 2000 + 101 * r_ix)
        anchors = anchor_rng.choice(candidates, size=need, replace=False)
        # Fixed region-level row generation makes overlapping rows exactly shared
        # across all four L scenarios.
        block_rng = np.random.default_rng(seed + 3000 + 101 * r_ix)
        master_blocks[region] = _crossed_l_region_block(
            data, anchors, supports[region], block_rng
        )

    scenarios: Dict[str, Scenario] = {}
    for s_ix, dominant_profile in enumerate(L_PROFILE_ORDER):
        dominant_region = next(
            r for r in X_REGIONS if str(profile_map[r]) == dominant_profile
        )
        alloc = allocations[dominant_profile]

        selected_anchor_parts: List[np.ndarray] = []
        source_parts: List[np.ndarray] = []
        t_parts: List[np.ndarray] = []
        y_clean_parts: List[np.ndarray] = []
        y_cont_parts: List[np.ndarray] = []
        for region in X_REGIONS:
            n_anchor = int(alloc[region])
            block = master_blocks[region]
            n_rows = n_anchor * (K_BITS + 1)
            selected_anchor_parts.append(block["anchors"][:n_anchor])
            source_parts.append(block["source_idx"][:n_rows])
            t_parts.append(block["t"][:n_rows])
            y_clean_parts.append(block["y_clean"][:n_rows])
            y_cont_parts.append(block["y_contaminated"][:n_rows])

        selected_anchors = np.concatenate(selected_anchor_parts)
        source_idx = np.concatenate(source_parts)
        t_override = np.concatenate(t_parts)
        y_clean = np.concatenate(y_clean_parts)
        y_contaminated = np.concatenate(y_cont_parts)

        # Shuffle expanded rows jointly while preserving the exact row identity
        # shared across matched scenarios.
        order = np.arange(len(source_idx))
        np.random.default_rng(seed + 9000 + s_ix).shuffle(order)
        source_idx = source_idx[order]
        t_override = t_override[order]
        y_clean = y_clean[order]
        y_contaminated = y_contaminated[order]

        name = L_SCENARIO_NAMES[dominant_profile]
        scenarios[name] = Scenario(
            name=name,
            train_idx=np.asarray(selected_anchors, dtype=int),
            test_idx=np.asarray(population_eval_idx, dtype=int),
            description=(
                "L controlled rank-4 norm-richness benchmark: every X region has "
                "origin plus four independent treatments (rank 4, |S|=5), with "
                "equal within-anchor representation of its five support vertices. "
                f"The {dominant_profile} region contributes {dominant_share:.0%} "
                "of training anchors; all 16 treatments are evaluated."
            ),
            local_supports={k: tuple(map(int, v)) for k, v in supports.items()},
            local_bases={k: tuple(map(int, v)) for k, v in bases.items()},
            family="L_RANK4_NORM_RICHNESS",
            metadata={
                "l_dominant_profile": dominant_profile,
                "l_dominant_region": dominant_region,
                "l_dominant_share": float(dominant_share),
                "l_profile_map": dict(profile_map),
                "l_weight_patterns": {
                    k: tuple(map(int, v)) for k, v in L_WEIGHT_PATTERNS.items()
                },
                "l_region_support_bits": {
                    region: bit_strings(supports[region]) for region in X_REGIONS
                },
                "l_region_basis_bits": {
                    region: bit_strings(bases[region]) for region in X_REGIONS
                },
                "l_anchor_allocation": dict(alloc),
                "l_n_anchors": int(total_anchors),
                "l_n_train_rows": int(len(source_idx)),
                "l_controlled_crossed_support": True,
                "l_all16_evaluation": True,
            },
            train_source_idx_override=source_idx,
            train_t_override=t_override,
            train_y_clean_override=y_clean,
            train_y_contaminated_override=y_contaminated,
        )

    return scenarios, {
        "profile_map": dict(profile_map),
        "supports": supports,
        "bases": bases,
        "weight_patterns": dict(L_WEIGHT_PATTERNS),
        "train_n": int(total_anchors * (K_BITS + 1)),
        "n_anchors": int(total_anchors),
        "dominant_share": float(dominant_share),
        "allocations": allocations,
        "controlled_crossed_support": True,
    }


SUPPORT_LADDER_ORDER = (
    "FULL16",
    "R4_REDUNDANT",
    "R4_CANONICAL",
    "R3_REDUNDANT",
)

SUPPORT_LADDER_SCENARIO_NAMES = {
    "FULL16": "H_LOCAL_SUPPORT_LADDER_FULL16_DOM",
    "R4_REDUNDANT": "H_LOCAL_SUPPORT_LADDER_R4_REDUNDANT_DOM",
    "R4_CANONICAL": "H_LOCAL_SUPPORT_LADDER_R4_CANONICAL_DOM",
    "R3_REDUNDANT": "H_LOCAL_SUPPORT_LADDER_R3_REDUNDANT_DOM",
}


def support_ladder_cap(
    data: DGPData,
    train_pool: np.ndarray,
    region_labels: np.ndarray,
    profile_supports: Mapping[str, Mapping[str, Sequence[int]]],
    profile_bases: Mapping[str, Mapping[str, Sequence[int]]],
    dominant_region: str,
    dominant_share: float,
    max_train_n: int | None,
) -> int:
    """Largest common n feasible for all four localized support-ladder scenarios."""
    cap = int(max_train_n or len(train_pool))
    cap = (cap // 20) * 20
    for total in range(cap, 399, -20):
        alloc = mixture_counts(total, dominant_region, dominant_share)
        feasible = True
        for profile in SUPPORT_LADDER_ORDER:
            supports = profile_supports[profile]
            bases = profile_bases[profile]
            for region, n_target in alloc.items():
                pool = candidate_pool(data, train_pool, region_labels, region, supports[region])
                if n_target > len(pool):
                    feasible = False
                    break
                counts = np.bincount(data.T[pool], minlength=N_REGIMES)
                min_protected = max(1, int(math.ceil(MIN_BASIS_EMPIRICAL_SHARE * n_target)))
                protected = set(map(int, bases[region])) | {ORIGIN}
                if any(counts[t] < min_protected for t in protected):
                    feasible = False
                    break
                if any(counts[t] < 1 for t in supports[region]):
                    feasible = False
                    break
                minimum_total = sum(
                    max(1, min_protected if t in protected else 1)
                    for t in set(map(int, supports[region]))
                )
                if minimum_total > n_target:
                    feasible = False
                    break
            if not feasible:
                break
        if feasible:
            return total
    raise RuntimeError("Could not find feasible H support-ladder n_train >=400.")


def choose_matched_h_redundant_vertex(
    data: DGPData,
    train_pool: np.ndarray,
    region_labels: np.ndarray,
    region: str,
    retained_basis: Sequence[int],
    rng: np.random.Generator,
) -> int:
    """Choose v in span(retained_basis), distinct from the canonical basis.

    The H ladder uses a matched pair:
      R4_CANONICAL = {0, retained_basis, omitted_direction}
      R3_REDUNDANT = {0, retained_basis, v}

    where v adds no new direction.  Among empirically available redundant
    vertices, prefer those with higher factual support in the dominant region;
    ties are randomized to avoid permanently privileging one interaction vertex.
    """
    retained_basis = tuple(map(int, retained_basis))
    if rank_of_codes(retained_basis) != K_BITS - 1:
        raise ValueError("H retained_basis must have rank K-1.")

    ids = train_pool[region_labels[train_pool] == region]
    counts = np.bincount(data.T[ids], minlength=N_REGIMES)
    canonical = set(map(int, CANONICAL_NONZERO))

    candidates: List[Tuple[int, int]] = []
    for code in range(1, N_REGIMES):
        if code in canonical or counts[code] <= 0:
            continue
        # Exact real-vector span test: adding v must not increase rank.
        if rank_of_codes((*retained_basis, int(code))) != K_BITS - 1:
            continue
        candidates.append((int(counts[code]), int(code)))

    if not candidates:
        raise RuntimeError(
            f"{region}: no empirically observed redundant vertex exists inside "
            f"span{tuple(bit_string(c) for c in retained_basis)}."
        )

    best_count = max(c for c, _ in candidates)
    # Keep only well-supported choices close to the best available count, then
    # randomize among them. This respects the weak-positivity emphasis on
    # non-negligible empirical support without outcome-based selection.
    eligible = [code for count, code in candidates if count >= 0.75 * best_count]
    eligible = sorted(eligible)
    return int(eligible[int(rng.integers(0, len(eligible)))])


def build_local_support_ladder_scenarios(
    data: DGPData,
    train_pool: np.ndarray,
    population_eval_idx: np.ndarray,
    region_labels: np.ndarray,
    rotation: int,
    seed: int,
    dominant_share: float,
    max_train_n: int | None,
    cycle_index: int | None = None,
) -> Tuple[Dict[str, Scenario], Dict[str, object]]:
    """
    Four-step localized, *matched* support ladder.

    The same physical X region is dominant in all four scenarios within a seed.
    The other three regions are held at full 16-vertex support and, importantly,
    use the exact same sampled observations across the four scenarios. Only the
    treatment support in the dominant region changes.

    Let e_m be one canonical direction selected for removal, let
    R={e_j:j!=m} be the three retained canonical directions, and choose one
    empirically observed binary vertex v such that v in span(R) but v is not a
    canonical singleton. Then the dominant-region supports are

      FULL16       = all 16 vertices;
      R4_REDUNDANT = {0, R, e_m, v};          rank 4, |S|=6;
      R4_CANONICAL = {0, R, e_m};             rank 4, |S|=5;
      R3_REDUNDANT = {0, R, v};               rank 3, |S|=5.

    Therefore R4_CANONICAL -> R3_REDUNDANT changes exactly one support vertex:
    an independent treatment direction e_m is replaced one-for-one by the same
    redundant vertex v already used in R4_REDUNDANT. Cardinality is fixed at 5.

    Across successful Monte-Carlo repetitions, the dominant X region cycles
    over Q1..Q4 and the omitted canonical direction is counterbalanced so that,
    over 16 successful repetitions, every physical region is paired with every
    omitted treatment component once.
    """
    dominant_region = X_REGIONS[int(rotation) % len(X_REGIONS)]
    full_support = tuple(range(N_REGIMES))
    full_basis = tuple(map(int, CANONICAL_NONZERO))

    if cycle_index is None:
        cycle_index = int(rotation)
    # Latin-style counterbalancing across 16 successful repetitions:
    # dominant region = cycle_index mod 4; omitted component shifts every block of 4.
    missing_component = (int(rotation) + int(cycle_index) // K_BITS) % K_BITS
    omitted_direction = int(CANONICAL_NONZERO[missing_component])
    retained_basis = tuple(
        int(code) for j, code in enumerate(CANONICAL_NONZERO) if j != missing_component
    )

    redundant_vertex = choose_matched_h_redundant_vertex(
        data,
        train_pool,
        region_labels,
        dominant_region,
        retained_basis,
        np.random.default_rng(seed + 101),
    )

    r4_red_support = tuple(sorted((ORIGIN, *full_basis, redundant_vertex)))
    r4_canonical_support = tuple(sorted((ORIGIN, *full_basis)))
    r3_red_support = tuple(sorted((ORIGIN, *retained_basis, redundant_vertex)))

    # Formal matched-geometry guards.
    if rank_of_codes(r4_red_support) != K_BITS:
        raise AssertionError("H R4_REDUNDANT failed rank K.")
    if rank_of_codes(r4_canonical_support) != K_BITS:
        raise AssertionError("H R4_CANONICAL failed rank K.")
    if rank_of_codes(r3_red_support) != K_BITS - 1:
        raise AssertionError("H R3_REDUNDANT failed rank K-1.")
    if rank_of_codes((*retained_basis, redundant_vertex)) != K_BITS - 1:
        raise AssertionError("H redundant vertex unexpectedly adds a treatment direction.")
    if not set(r4_canonical_support).issubset(set(r4_red_support)):
        raise AssertionError("H R4_REDUNDANT must contain the full canonical support.")
    if not set(r3_red_support).issubset(set(r4_red_support)):
        raise AssertionError("H R3_REDUNDANT must be nested inside R4_REDUNDANT.")
    removed = set(r4_canonical_support) - set(r3_red_support)
    added = set(r3_red_support) - set(r4_canonical_support)
    if removed != {omitted_direction} or added != {redundant_vertex}:
        raise AssertionError(
            "H matched rank contrast must swap exactly one independent direction "
            "for exactly one redundant vertex."
        )

    dominant_support = {
        "FULL16": full_support,
        "R4_REDUNDANT": r4_red_support,
        "R4_CANONICAL": r4_canonical_support,
        "R3_REDUNDANT": r3_red_support,
    }
    dominant_basis = {
        "FULL16": full_basis,
        "R4_REDUNDANT": full_basis,
        "R4_CANONICAL": full_basis,
        "R3_REDUNDANT": retained_basis,
    }

    profile_supports: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    profile_bases: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    for profile in SUPPORT_LADDER_ORDER:
        profile_supports[profile] = {
            region: (
                tuple(map(int, dominant_support[profile]))
                if region == dominant_region
                else full_support
            )
            for region in X_REGIONS
        }
        profile_bases[profile] = {
            region: (
                tuple(map(int, dominant_basis[profile]))
                if region == dominant_region
                else full_basis
            )
            for region in X_REGIONS
        }

    total_n = support_ladder_cap(
        data,
        train_pool,
        region_labels,
        profile_supports,
        profile_bases,
        dominant_region,
        dominant_share,
        max_train_n,
    )
    alloc = mixture_counts(total_n, dominant_region, dominant_share)

    # Hold the three non-dominant regions literally fixed across all four H scenarios.
    shared_minor_ids: Dict[str, np.ndarray] = {}
    for r_ix, region in enumerate(X_REGIONS):
        if region == dominant_region:
            continue
        pool = candidate_pool(data, train_pool, region_labels, region, full_support)
        shared_minor_ids[region] = sample_with_minimum_support(
            pool,
            data.T,
            full_support,
            full_basis,
            alloc[region],
            np.random.default_rng(seed + 1000 + 101 * r_ix),
            require_all_support=True,
        )

    scenarios: Dict[str, Scenario] = {}
    for p_ix, profile in enumerate(SUPPORT_LADDER_ORDER):
        supports = profile_supports[profile]
        bases = profile_bases[profile]
        pool = candidate_pool(
            data, train_pool, region_labels, dominant_region, supports[dominant_region]
        )
        dominant_ids = sample_with_minimum_support(
            pool,
            data.T,
            supports[dominant_region],
            bases[dominant_region],
            alloc[dominant_region],
            np.random.default_rng(seed + 5000 + 1000 * p_ix),
            require_all_support=True,
        )
        pieces = [
            dominant_ids if region == dominant_region else shared_minor_ids[region]
            for region in X_REGIONS
        ]
        tr = np.concatenate(pieces)
        np.random.default_rng(seed + 9000 + p_ix).shuffle(tr)

        dom_geom = geometry_diagnostics(supports[dominant_region])
        name = SUPPORT_LADDER_SCENARIO_NAMES[profile]
        scenarios[name] = Scenario(
            name=name,
            train_idx=tr,
            test_idx=np.asarray(population_eval_idx, dtype=int),
            description=(
                "Matched localized support ladder with the same dominant X region "
                f"contributing {dominant_share:.0%} of training data; "
                f"dominant-region profile={profile}. All three non-dominant X "
                "regions retain full 16-vertex support."
            ),
            local_supports={k: tuple(map(int, v)) for k, v in supports.items()},
            local_bases={k: tuple(map(int, v)) for k, v in bases.items()},
            family="H_LOCAL_SUPPORT_LADDER",
            metadata={
                "dominant_region": dominant_region,
                "dominant_profile": profile,
                "dominant_rank": int(dom_geom["rank_U"]),
                "dominant_n_support_vertices": int(dom_geom["n_support_vertices"]),
                "dominant_share": float(dominant_share),
                "missing_component_index": int(missing_component),
                "omitted_direction_code": int(omitted_direction),
                "omitted_direction_bits": bit_string(omitted_direction),
                "redundant_vertex_code": int(redundant_vertex),
                "redundant_vertex_bits": bit_string(redundant_vertex),
                "retained_basis_codes": tuple(map(int, retained_basis)),
                "retained_basis_bits": tuple(bit_string(c) for c in retained_basis),
            },
        )

    return scenarios, {
        "dominant_region": dominant_region,
        "dominant_share": dominant_share,
        "train_n": total_n,
        "allocation": alloc,
        "profile_supports": profile_supports,
        "profile_bases": profile_bases,
        "missing_component_index": int(missing_component),
        "omitted_direction_code": int(omitted_direction),
        "omitted_direction_bits": bit_string(omitted_direction),
        "retained_basis": retained_basis,
        "retained_basis_bits": tuple(bit_string(c) for c in retained_basis),
        "redundant_vertex": int(redundant_vertex),
        "redundant_vertex_bits": bit_string(redundant_vertex),
        "r4_redundant_support": r4_red_support,
        "r4_canonical_support": r4_canonical_support,
        "r3_redundant_support": r3_red_support,
        "rotation": rotation,
        "cycle_index": int(cycle_index),
    }



J_SUPPORT_SMOOTHING_ORDER = (
    "FULL16",
    "SUPPORT12",
    "SUPPORT8",
    "CANONICAL4_PLUS_ORIGIN",
)

J_SUPPORT_SMOOTHING_SCENARIO_NAMES = {
    # Endpoints are deliberately the H scenarios; only the middle two are new fits.
    "FULL16": SUPPORT_LADDER_SCENARIO_NAMES["FULL16"],
    "SUPPORT12": "J_RANK4_SUPPORT12_DOM",
    "SUPPORT8": "J_RANK4_SUPPORT8_DOM",
    "CANONICAL4_PLUS_ORIGIN": SUPPORT_LADDER_SCENARIO_NAMES["R4_CANONICAL"],
}


def choose_nested_j_joint_vertices(seed: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Choose nested noncanonical joint vertices for the J rank-4 ladder.

    For K=4 there are 11 noncanonical nonzero joint vertices.  The canonical
    directions are retained at every rung, so rank is always exactly four.
    Selection is uniform over the 11 joint vertices and never uses outcomes.
    SUPPORT12 keeps seven joints; SUPPORT8 keeps the first three of those seven.
    Re-randomization across seeds prevents one particular interaction geometry
    from defining the experiment.
    """
    canonical = set(map(int, CANONICAL_NONZERO))
    joint = np.asarray(
        [c for c in range(1, N_REGIMES) if c not in canonical], dtype=int
    )
    if len(joint) != 11:
        raise AssertionError("K=4 must have exactly 11 noncanonical nonzero vertices.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(joint)
    keep7 = tuple(sorted(map(int, perm[:7])))
    keep3 = tuple(sorted(map(int, perm[:3])))
    if not set(keep3).issubset(set(keep7)):
        raise AssertionError("J SUPPORT8 joints must be nested inside SUPPORT12 joints.")
    return keep7, keep3


def build_rank4_support_smoothing_scenarios(
    data: DGPData,
    train_pool: np.ndarray,
    population_eval_idx: np.ndarray,
    region_labels: np.ndarray,
    seed: int,
    h_scenarios: Mapping[str, Scenario],
    h_meta: Mapping[str, object],
) -> Tuple[Dict[str, Scenario], Dict[str, object]]:
    """Construct J: a nested full-rank support-cardinality ladder.

    J is intentionally matched to H.  FULL16 and CANONICAL are not refit here:
    they are exactly H_LOCAL_SUPPORT_LADDER_FULL16_DOM and
    H_LOCAL_SUPPORT_LADDER_R4_CANONICAL_DOM.  We add only SUPPORT12 and SUPPORT8.

    In the dominant X region:
      FULL16    = {0} + 15 nonzero vertices                    |S|=16, rank 4
      SUPPORT12 = {0,e1,e2,e3,e4} + 7 joint vertices          |S|=12, rank 4
      SUPPORT8  = {0,e1,e2,e3,e4} + 3 of those joint vertices |S|= 8, rank 4
      CANONICAL = {0,e1,e2,e3,e4}                              |S|= 5, rank 4

    The three non-dominant regions use the exact same observations as H and keep
    full 16-vertex support.  The total n and the 70/10/10/10 (or user-specified)
    allocation are inherited from H, so the J path isolates treatment-support
    richness rather than sample size, X composition, or rank.
    """
    dominant_region = str(h_meta["dominant_region"])
    full_support = tuple(range(N_REGIMES))
    canonical_basis = tuple(map(int, CANONICAL_NONZERO))
    canonical_support = tuple(sorted((ORIGIN, *canonical_basis)))

    full_name = SUPPORT_LADDER_SCENARIO_NAMES["FULL16"]
    canonical_name = SUPPORT_LADDER_SCENARIO_NAMES["R4_CANONICAL"]
    if full_name not in h_scenarios or canonical_name not in h_scenarios:
        raise RuntimeError("J requires the matched H FULL16 and R4_CANONICAL endpoints.")
    h_full = h_scenarios[full_name]
    h_can = h_scenarios[canonical_name]

    total_n = int(h_meta["train_n"])
    alloc = {str(k): int(v) for k, v in dict(h_meta["allocation"]).items()}
    if sum(alloc.values()) != total_n:
        raise AssertionError("J inherited H allocation does not sum to H train_n.")

    # Verify the endpoints really are the intended matched H supports.
    if tuple(sorted(h_full.local_supports[dominant_region])) != full_support:
        raise AssertionError("J/H FULL16 endpoint mismatch.")
    if tuple(sorted(h_can.local_supports[dominant_region])) != canonical_support:
        raise AssertionError("J/H canonical endpoint mismatch.")

    keep7, keep3 = choose_nested_j_joint_vertices(seed + 313)
    support12 = tuple(sorted((ORIGIN, *canonical_basis, *keep7)))
    support8 = tuple(sorted((ORIGIN, *canonical_basis, *keep3)))

    expected = [(support12, 12), (support8, 8), (canonical_support, 5)]
    for support, size in expected:
        if len(support) != size:
            raise AssertionError(f"J support cardinality mismatch: got {len(support)}, expected {size}.")
        if rank_of_codes(support) != K_BITS:
            raise AssertionError("Every J rung must retain full treatment rank K=4.")
        if not set(canonical_support).issubset(set(support)):
            raise AssertionError("Every J rung must contain the canonical basis and origin.")
    if not set(support8).issubset(set(support12)):
        raise AssertionError("J SUPPORT8 must be nested inside SUPPORT12.")
    if not set(support12).issubset(set(full_support)):
        raise AssertionError("J SUPPORT12 must be nested inside FULL16.")

    # Reuse exactly the non-dominant H observations, not merely the same mixture.
    h_full_idx = np.asarray(h_full.train_idx, dtype=int)
    shared_minor_ids: Dict[str, np.ndarray] = {}
    for region in X_REGIONS:
        if region == dominant_region:
            continue
        ids = h_full_idx[region_labels[h_full_idx] == region]
        if len(ids) != alloc[region]:
            raise AssertionError(
                f"J expected {alloc[region]} inherited H rows in {region}, found {len(ids)}."
            )
        shared_minor_ids[region] = ids.copy()

    middle_profiles = {
        "SUPPORT12": support12,
        "SUPPORT8": support8,
    }
    scenarios: Dict[str, Scenario] = {}
    for p_ix, (profile, dom_support) in enumerate(middle_profiles.items()):
        dom_pool = candidate_pool(
            data, train_pool, region_labels, dominant_region, dom_support
        )
        dominant_ids = sample_with_minimum_support(
            dom_pool,
            data.T,
            dom_support,
            canonical_basis,
            alloc[dominant_region],
            np.random.default_rng(seed + 15_000 + 1000 * p_ix),
            require_all_support=True,
        )
        pieces = [
            dominant_ids if region == dominant_region else shared_minor_ids[region]
            for region in X_REGIONS
        ]
        tr = np.concatenate(pieces)
        np.random.default_rng(seed + 18_000 + p_ix).shuffle(tr)

        name = J_SUPPORT_SMOOTHING_SCENARIO_NAMES[profile]
        local_supports = {
            region: (tuple(dom_support) if region == dominant_region else full_support)
            for region in X_REGIONS
        }
        local_bases = {region: canonical_basis for region in X_REGIONS}
        scenarios[name] = Scenario(
            name=name,
            train_idx=tr,
            test_idx=np.asarray(population_eval_idx, dtype=int),
            description=(
                "J nested rank-4 support smoothing in the same dominant X region as H; "
                f"profile={profile}, dominant support has {len(dom_support)} total vertices. "
                "All canonical directions remain observed and the other three X regions "
                "reuse H's full-support observations exactly."
            ),
            local_supports=local_supports,
            local_bases=local_bases,
            family="J_RANK4_SUPPORT_SMOOTHING",
            metadata={
                "dominant_region": dominant_region,
                "dominant_profile": profile,
                "dominant_rank": K_BITS,
                "dominant_n_support_vertices": int(len(dom_support)),
                "dominant_share": float(h_meta["dominant_share"]),
                "support_rank": K_BITS,
                "support_cardinality": int(len(dom_support)),
                "j_total_support_vertices": int(len(dom_support)),
                "j_nonzero_support_vertices": int(len(dom_support) - 1),
                "j_extra_joint_vertices": int(len(dom_support) - 1 - K_BITS),
                "j_joint_vertices_bits": tuple(
                    bit_string(c) for c in dom_support
                    if c != ORIGIN and c not in set(canonical_basis)
                ),
                "j_nested_support12_bits": tuple(bit_string(c) for c in support12),
                "j_nested_support8_bits": tuple(bit_string(c) for c in support8),
            },
        )

    return scenarios, {
        "dominant_region": dominant_region,
        "dominant_share": float(h_meta["dominant_share"]),
        "train_n": total_n,
        "allocation": alloc,
        "full16_scenario": full_name,
        "support12_scenario": J_SUPPORT_SMOOTHING_SCENARIO_NAMES["SUPPORT12"],
        "support8_scenario": J_SUPPORT_SMOOTHING_SCENARIO_NAMES["SUPPORT8"],
        "canonical_scenario": canonical_name,
        "support16": full_support,
        "support12": support12,
        "support8": support8,
        "support5_canonical": canonical_support,
        "support12_bits": tuple(bit_string(c) for c in support12),
        "support8_bits": tuple(bit_string(c) for c in support8),
        "canonical_bits": tuple(bit_string(c) for c in canonical_support),
        "joint_keep7_bits": tuple(bit_string(c) for c in keep7),
        "joint_keep3_bits": tuple(bit_string(c) for c in keep3),
    }



K_NEIGHBOR_SCENARIO_NAMES = {
    "FULL16_REFERENCE": "K_NEIGHBOR_FULL16_REFERENCE",
    "HALF_CANONICAL": "K_NEIGHBOR_HALF_CANONICAL",
}


def build_neighbor_support_extrapolation_scenarios(
    data: DGPData,
    full2k_scenario: Scenario,
    region_labels: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, Scenario], Dict[str, object]]:
    """Construct K: interleaved neighboring X populations with different T support.

    Start from the exact synthetic rows of BASE_FULL_2K_GIVEN_X.  Inside each
    X stratum, order the baseline anchors by S_extra and make consecutive pairs.
    Position 0 of each pair is the weak target X and position 1 is its full-
    support neighbor.  To counterbalance edge direction across repetitions, the
    order is reversed on odd seeds, while the weak anchor remains the even
    position in the ordered vector.

    FULL16_REFERENCE keeps all 16 rows for every selected anchor.
    HALF_CANONICAL deletes every noncanonical row from the weak/even anchors but
    leaves the neighboring full anchors untouched.  Retained rows reuse exactly
    the same Bernoulli outcomes generated for the full-2^K baseline; no outcomes
    are redrawn.  Therefore the treatment-row deletion is the only intervention.

    Both scenarios are evaluated on the same weak/even X anchors over all 16
    actions.  The 11 noncanonical actions at those X values are directly observed
    in the reference and absent in HALF_CANONICAL, so their prediction must be
    borrowed from interleaved full-support neighbors in X-space.
    """
    if not full2k_scenario.has_training_override():
        raise ValueError("K requires the literal full-2^K|x baseline overrides.")

    full_support = tuple(range(N_REGIMES))
    canonical_support = tuple(sorted((ORIGIN, *CANONICAL_NONZERO)))
    canonical_set = set(canonical_support)

    base_anchors = np.asarray(full2k_scenario.train_idx, dtype=int)
    weak_parts: List[np.ndarray] = []
    full_parts: List[np.ndarray] = []
    selected_parts: List[np.ndarray] = []
    paired_weak: List[int] = []
    paired_full: List[int] = []
    region_counts: Dict[str, Dict[str, int]] = {}

    descending = bool(int(seed) % 2)
    for region in X_REGIONS:
        anchors = base_anchors[region_labels[base_anchors] == region]
        if len(anchors) < 2:
            raise RuntimeError(f"K needs at least two full-2^K anchors in {region}.")
        order = np.argsort(data.s_extra[anchors], kind="mergesort")
        ordered = anchors[order]
        if descending:
            ordered = ordered[::-1]
        # Exact half/half split.  Drop at most one endpoint when the region has
        # an odd number of baseline anchors; which endpoint is dropped flips
        # with ordering direction across repetitions.
        n_even = 2 * (len(ordered) // 2)
        ordered = ordered[:n_even]
        if len(ordered) < 2:
            raise RuntimeError(f"K has insufficient paired anchors in {region}.")

        weak = np.asarray(ordered[0::2], dtype=int)
        full = np.asarray(ordered[1::2], dtype=int)
        if len(weak) != len(full):
            raise AssertionError("K alternating split must produce equal halves.")

        weak_parts.append(weak)
        full_parts.append(full)
        selected_parts.append(np.asarray(ordered, dtype=int))
        paired_weak.extend(map(int, weak))
        paired_full.extend(map(int, full))
        region_counts[region] = {
            "n_selected": int(len(ordered)),
            "n_weak": int(len(weak)),
            "n_full": int(len(full)),
        }

    weak_idx = np.concatenate(weak_parts)
    full_idx = np.concatenate(full_parts)
    selected_idx = np.concatenate(selected_parts)
    if len(np.intersect1d(weak_idx, full_idx)):
        raise AssertionError("K weak/full X populations must be disjoint.")
    if len(weak_idx) != len(full_idx):
        raise AssertionError("K weak/full X populations must have equal size globally.")

    # Pair diagnostics: adjacent in the S_extra ordering, plus distance in the
    # standardized structural X space.  These quantify that K is a local-neighbor
    # experiment rather than a missing-X-region experiment.
    weak_pair = np.asarray(paired_weak, dtype=int)
    full_pair = np.asarray(paired_full, dtype=int)
    # Adjacent parity pairs are neighbors along the scalar S_extra ordering.
    ds = np.abs(data.s_extra[weak_pair] - data.s_extra[full_pair])

    # Also measure the actual nearest full-support neighbor in the standardized
    # 12-dimensional structural X space.  Membership of the full/weak halves is
    # still determined only by ordering parity; this diagnostic simply verifies
    # that a close full-support X is available to each weak target X.
    nearest_dx_parts: List[np.ndarray] = []
    for weak_r, full_r in zip(weak_parts, full_parts):
        dmat = cdist(data.X_star[weak_r], data.X_star[full_r], metric="euclidean")
        nearest_dx_parts.append(np.min(dmat, axis=1))
    dx = np.concatenate(nearest_dx_parts)

    base_src = full2k_scenario.training_source_idx()
    base_t = full2k_scenario.training_t(data)
    base_y_clean = np.asarray(full2k_scenario.train_y_clean_override, dtype=float)
    base_y_cont = np.asarray(full2k_scenario.train_y_contaminated_override, dtype=float)

    selected_mask = np.isin(base_src, selected_idx)
    ref_src = base_src[selected_mask]
    ref_t = base_t[selected_mask]
    ref_y_clean = base_y_clean[selected_mask]
    ref_y_cont = base_y_cont[selected_mask]

    weak_row = np.isin(ref_src, weak_idx)
    keep_mixed = (~weak_row) | np.isin(ref_t, np.asarray(canonical_support, dtype=int))
    mix_src = ref_src[keep_mixed]
    mix_t = ref_t[keep_mixed]
    mix_y_clean = ref_y_clean[keep_mixed]
    mix_y_cont = ref_y_cont[keep_mixed]

    def _check_per_x(src: np.ndarray, t: np.ndarray, mixed: bool) -> None:
        for x in selected_idx:
            obs = tuple(sorted(np.unique(t[src == x]).astype(int).tolist()))
            expected = canonical_support if (mixed and x in set(map(int, weak_idx))) else full_support
            if obs != expected:
                raise AssertionError(
                    f"K support mismatch at x={int(x)}: got {bit_strings(obs)}, "
                    f"expected {bit_strings(expected)}."
                )

    _check_per_x(ref_src, ref_t, mixed=False)
    _check_per_x(mix_src, mix_t, mixed=True)

    common_meta = {
        "evaluation_mode": "neighbor_weak_x_all16",
        "k_n_weak_x": int(len(weak_idx)),
        "k_n_full_x": int(len(full_idx)),
        "k_weak_anchor_idx": tuple(map(int, weak_idx)),
        "k_full_anchor_idx": tuple(map(int, full_idx)),
        "k_selected_anchor_idx": tuple(map(int, selected_idx)),
        "k_weak_support_codes": canonical_support,
        "k_weak_support_bits": tuple(bit_string(c) for c in canonical_support),
        "k_full_support_codes": full_support,
        "k_full_support_bits": tuple(bit_string(c) for c in full_support),
        "k_noncanonical_codes": tuple(c for c in full_support if c not in canonical_set),
        "k_order_direction": "descending" if descending else "ascending",
        "k_neighbor_s_extra_mean": float(np.mean(ds)),
        "k_neighbor_s_extra_median": float(np.median(ds)),
        "k_neighbor_s_extra_p95": float(np.quantile(ds, 0.95)),
        "k_neighbor_xstar_l2_mean": float(np.mean(dx)),
        "k_neighbor_xstar_l2_median": float(np.median(dx)),
        "k_neighbor_xstar_l2_p95": float(np.quantile(dx, 0.95)),
        "k_neighbor_xstar_definition": "nearest full-support anchor within same X region",
        "k_region_counts": region_counts,
    }

    ref = Scenario(
        name=K_NEIGHBOR_SCENARIO_NAMES["FULL16_REFERENCE"],
        train_idx=np.asarray(selected_idx, dtype=int),
        test_idx=np.asarray(weak_idx, dtype=int),
        description=(
            "K neighbor reference: the same interleaved X anchors used by the weak-half "
            "scenario, with all 16 treatments retained at every anchor. Evaluation is "
            "on the weak/even anchor population over all 16 treatments."
        ),
        local_supports={r: full_support for r in X_REGIONS},
        local_bases={r: tuple(CANONICAL_NONZERO) for r in X_REGIONS},
        target_kind="neighbor_weak_x",
        family="K_NEIGHBOR_SUPPORT_EXTRAPOLATION",
        metadata={
            **common_meta,
            "k_profile": "FULL16_REFERENCE",
            "literal_all_2k_per_x": 1,
        },
        train_source_idx_override=np.asarray(ref_src, dtype=int),
        train_t_override=np.asarray(ref_t, dtype=int),
        train_y_clean_override=np.asarray(ref_y_clean, dtype=float),
        train_y_contaminated_override=np.asarray(ref_y_cont, dtype=float),
    )

    mixed = Scenario(
        name=K_NEIGHBOR_SCENARIO_NAMES["HALF_CANONICAL"],
        train_idx=np.asarray(selected_idx, dtype=int),
        test_idx=np.asarray(weak_idx, dtype=int),
        description=(
            "K neighbor extrapolation: X anchors are interleaved within every region; "
            "the even/weak half retains only origin plus the canonical rank-4 basis, "
            "while each adjacent odd/full neighbor retains all 16 treatments. Evaluation "
            "asks for all 16 outcomes at the weak X anchors."
        ),
        # Region-level union is full support, but metadata records the exact per-X split.
        local_supports={r: full_support for r in X_REGIONS},
        local_bases={r: tuple(CANONICAL_NONZERO) for r in X_REGIONS},
        target_kind="neighbor_weak_x",
        family="K_NEIGHBOR_SUPPORT_EXTRAPOLATION",
        metadata={
            **common_meta,
            "k_profile": "HALF_CANONICAL",
            "literal_all_2k_per_x": 0,
        },
        train_source_idx_override=np.asarray(mix_src, dtype=int),
        train_t_override=np.asarray(mix_t, dtype=int),
        train_y_clean_override=np.asarray(mix_y_clean, dtype=float),
        train_y_contaminated_override=np.asarray(mix_y_cont, dtype=float),
    )

    return {ref.name: ref, mixed.name: mixed}, {
        **common_meta,
        "reference_scenario": ref.name,
        "mixed_scenario": mixed.name,
        "reference_train_rows": int(ref.n_training_rows()),
        "mixed_train_rows": int(mixed.n_training_rows()),
        "removed_rows": int(ref.n_training_rows() - mixed.n_training_rows()),
        "removed_fraction": float(1.0 - mixed.n_training_rows() / ref.n_training_rows()),
    }


K_UNSUPPORTED_SCENARIO_NAMES = {
    "FULL16_REFERENCE": "K_UNSUPPORTED_FULL16_REFERENCE",
    "MOD5": "K_UNSUPPORTED_MOD5",
}

K_UNSUPPORTED_GROUPS = (
    ("full", 0, tuple(range(N_REGIMES)), 4),
    ("r4", 1, CANONICAL_BY_RANK[4], 4),
    ("r3", 2, CANONICAL_BY_RANK[3], 3),
    ("r2", 3, CANONICAL_BY_RANK[2], 2),
    ("r1", 4, CANONICAL_BY_RANK[1], 1),
)


def build_k_unsupported_mod5_scenarios(
    data: DGPData,
    full2k_scenario: Scenario,
    region_labels: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, Scenario], Dict[str, object]]:
    """Construct K-UNSUPPORTED: five interleaved X populations with ranks 4..1.

    Start from the exact synthetic rows in BASE_FULL_2K_GIVEN_X.  Within each
    X region, sort anchors by S_extra, trim at most four endpoints so the count
    is divisible by five, and assign each ordered anchor by position modulo 5:

        0 -> full 16-vertex support
        1 -> canonical rank-4 support
        2 -> canonical rank-3 support
        3 -> canonical rank-2 support
        4 -> canonical rank-1 support

    The full reference and mixed MOD5 scenario use the same X anchors and reuse
    exactly the same realized Bernoulli outcome rows whenever retained.  All 16
    potential outcomes are evaluated separately for every modulo-5 X group, so
    unsupported-treatment error can be measured without a missing-X-region
    design: every support profile is interleaved across all four X strata.
    """
    if not full2k_scenario.has_training_override():
        raise ValueError("K-UNSUPPORTED requires the literal full-2^K|x baseline overrides.")

    full_support = tuple(range(N_REGIMES))
    support_by_label = {label: tuple(map(int, support)) for label, _, support, _ in K_UNSUPPORTED_GROUPS}
    rank_by_label = {label: int(rank) for label, _, _, rank in K_UNSUPPORTED_GROUPS}
    mod_by_label = {label: int(mod) for label, mod, _, _ in K_UNSUPPORTED_GROUPS}

    base_anchors = np.asarray(full2k_scenario.train_idx, dtype=int)
    group_parts: Dict[str, List[np.ndarray]] = {label: [] for label, *_ in K_UNSUPPORTED_GROUPS}
    selected_parts: List[np.ndarray] = []
    region_counts: Dict[str, Dict[str, int]] = {}

    descending = bool(int(seed) % 2)
    for region in X_REGIONS:
        anchors = base_anchors[region_labels[base_anchors] == region]
        if len(anchors) < 5:
            raise RuntimeError(f"K-UNSUPPORTED needs at least five full-2^K anchors in {region}.")
        order = np.argsort(data.s_extra[anchors], kind="mergesort")
        ordered = anchors[order]
        if descending:
            ordered = ordered[::-1]

        n5 = 5 * (len(ordered) // 5)
        ordered = np.asarray(ordered[:n5], dtype=int)
        if len(ordered) < 5:
            raise RuntimeError(f"K-UNSUPPORTED has insufficient anchors in {region} after trimming.")
        selected_parts.append(ordered)

        counts = {"n_selected": int(len(ordered))}
        for label, mod, _, _ in K_UNSUPPORTED_GROUPS:
            grp = np.asarray(ordered[int(mod)::5], dtype=int)
            group_parts[label].append(grp)
            counts[f"n_{label}"] = int(len(grp))
        if len({counts[f"n_{label}"] for label, *_ in K_UNSUPPORTED_GROUPS}) != 1:
            raise AssertionError("K-UNSUPPORTED modulo groups must be equal within each region.")
        region_counts[region] = counts

    group_idx = {label: np.concatenate(parts) for label, parts in group_parts.items()}
    selected_idx = np.concatenate(selected_parts)
    group_sizes = {label: int(len(idx)) for label, idx in group_idx.items()}
    if len(set(group_sizes.values())) != 1:
        raise AssertionError("K-UNSUPPORTED modulo groups must have equal global size.")

    # Verify that every selected anchor belongs to exactly one modulo group.
    stacked = np.concatenate(list(group_idx.values()))
    if len(np.unique(stacked)) != len(stacked) or set(map(int, stacked)) != set(map(int, selected_idx)):
        raise AssertionError("K-UNSUPPORTED modulo partition is not an exact partition of selected anchors.")

    # Distance diagnostics: every non-full anchor should have regularly interleaved
    # full-support representatives in the same physical X region.  Membership is
    # still defined solely by ordered position modulo 5; nearest-neighbor distance
    # is a diagnostic, not a selection criterion.
    nonfull_idx = np.concatenate([group_idx[label] for label in ("r4", "r3", "r2", "r1")])
    nearest_ds_parts: List[np.ndarray] = []
    nearest_dx_parts: List[np.ndarray] = []
    for region in X_REGIONS:
        full_r = group_idx["full"][region_labels[group_idx["full"]] == region]
        nonfull_r = nonfull_idx[region_labels[nonfull_idx] == region]
        if not len(full_r) or not len(nonfull_r):
            raise RuntimeError(f"K-UNSUPPORTED missing full/nonfull anchors in {region}.")
        ds_mat = np.abs(data.s_extra[nonfull_r, None] - data.s_extra[full_r][None, :])
        nearest_ds_parts.append(np.min(ds_mat, axis=1))
        dmat = cdist(data.X_star[nonfull_r], data.X_star[full_r], metric="euclidean")
        nearest_dx_parts.append(np.min(dmat, axis=1))
    nearest_ds = np.concatenate(nearest_ds_parts)
    nearest_dx = np.concatenate(nearest_dx_parts)

    base_src = full2k_scenario.training_source_idx()
    base_t = full2k_scenario.training_t(data)
    base_y_clean = np.asarray(full2k_scenario.train_y_clean_override, dtype=float)
    base_y_cont = np.asarray(full2k_scenario.train_y_contaminated_override, dtype=float)

    selected_mask = np.isin(base_src, selected_idx)
    ref_src = base_src[selected_mask]
    ref_t = base_t[selected_mask]
    ref_y_clean = base_y_clean[selected_mask]
    ref_y_cont = base_y_cont[selected_mask]

    anchor_label: Dict[int, str] = {}
    for label, idx in group_idx.items():
        anchor_label.update({int(x): label for x in idx})

    keep_mixed = np.zeros(len(ref_src), dtype=bool)
    for i, (x, t) in enumerate(zip(ref_src, ref_t)):
        label = anchor_label[int(x)]
        keep_mixed[i] = int(t) in set(support_by_label[label])
    mix_src = ref_src[keep_mixed]
    mix_t = ref_t[keep_mixed]
    mix_y_clean = ref_y_clean[keep_mixed]
    mix_y_cont = ref_y_cont[keep_mixed]

    # Exact per-X verification in both designs.
    for label, idx in group_idx.items():
        expected_mixed = tuple(sorted(support_by_label[label]))
        for x in idx:
            ref_obs = tuple(sorted(np.unique(ref_t[ref_src == x]).astype(int).tolist()))
            mix_obs = tuple(sorted(np.unique(mix_t[mix_src == x]).astype(int).tolist()))
            if ref_obs != full_support:
                raise AssertionError(f"K-UNSUPPORTED reference lost treatment rows at x={int(x)}.")
            if mix_obs != expected_mixed:
                raise AssertionError(
                    f"K-UNSUPPORTED support mismatch at x={int(x)} ({label}): "
                    f"got {bit_strings(mix_obs)}, expected {bit_strings(expected_mixed)}."
                )
            if rank_of_codes(mix_obs) != rank_by_label[label]:
                raise AssertionError(
                    f"K-UNSUPPORTED rank mismatch at x={int(x)} ({label}): "
                    f"got {rank_of_codes(mix_obs)}, expected {rank_by_label[label]}."
                )

    common_meta = {
        "evaluation_mode": "mod5_groups_all16",
        "ku_selected_anchor_idx": tuple(map(int, selected_idx)),
        "ku_group_anchor_idx": {k: tuple(map(int, v)) for k, v in group_idx.items()},
        "ku_support_codes": support_by_label,
        "ku_support_bits": {k: tuple(bit_string(c) for c in v) for k, v in support_by_label.items()},
        "ku_rank_by_group": rank_by_label,
        "ku_mod_by_group": mod_by_label,
        "ku_group_sizes": group_sizes,
        "ku_order_direction": "descending" if descending else "ascending",
        "ku_neighbor_s_extra_mean": float(np.mean(nearest_ds)),
        "ku_neighbor_s_extra_median": float(np.median(nearest_ds)),
        "ku_neighbor_s_extra_p95": float(np.quantile(nearest_ds, 0.95)),
        "ku_neighbor_xstar_l2_mean": float(np.mean(nearest_dx)),
        "ku_neighbor_xstar_l2_median": float(np.median(nearest_dx)),
        "ku_neighbor_xstar_l2_p95": float(np.quantile(nearest_dx, 0.95)),
        "ku_neighbor_definition": "nearest mod-0 full-support anchor within same X region",
        "ku_region_counts": region_counts,
    }

    ref = Scenario(
        name=K_UNSUPPORTED_SCENARIO_NAMES["FULL16_REFERENCE"],
        train_idx=np.asarray(selected_idx, dtype=int),
        test_idx=np.asarray(selected_idx, dtype=int),
        description=(
            "K-UNSUPPORTED matched reference: the modulo-5 X populations are fixed, "
            "but every selected X anchor retains all 16 treatment vertices."
        ),
        local_supports={r: full_support for r in X_REGIONS},
        local_bases={r: tuple(CANONICAL_NONZERO) for r in X_REGIONS},
        target_kind="neighbor_mod5_x",
        family="K_UNSUPPORTED_MOD5_EXTRAPOLATION",
        metadata={**common_meta, "ku_profile": "FULL16_REFERENCE", "literal_all_2k_per_x": 1},
        train_source_idx_override=np.asarray(ref_src, dtype=int),
        train_t_override=np.asarray(ref_t, dtype=int),
        train_y_clean_override=np.asarray(ref_y_clean, dtype=float),
        train_y_contaminated_override=np.asarray(ref_y_cont, dtype=float),
    )

    mixed = Scenario(
        name=K_UNSUPPORTED_SCENARIO_NAMES["MOD5"],
        train_idx=np.asarray(selected_idx, dtype=int),
        test_idx=np.asarray(selected_idx, dtype=int),
        description=(
            "K-UNSUPPORTED modulo-5 extrapolation: after ordering X inside each region, "
            "positions 0,1,2,3,4 mod 5 receive full16, canonical rank4, rank3, rank2, "
            "and rank1 support respectively; all 16 outcomes are evaluated in each group."
        ),
        # Region-level union remains full because every region contains mod-0 full anchors.
        local_supports={r: full_support for r in X_REGIONS},
        local_bases={r: tuple(CANONICAL_NONZERO) for r in X_REGIONS},
        target_kind="neighbor_mod5_x",
        family="K_UNSUPPORTED_MOD5_EXTRAPOLATION",
        metadata={**common_meta, "ku_profile": "MOD5", "literal_all_2k_per_x": 0},
        train_source_idx_override=np.asarray(mix_src, dtype=int),
        train_t_override=np.asarray(mix_t, dtype=int),
        train_y_clean_override=np.asarray(mix_y_clean, dtype=float),
        train_y_contaminated_override=np.asarray(mix_y_cont, dtype=float),
    )

    return {ref.name: ref, mixed.name: mixed}, {
        **common_meta,
        "reference_scenario": ref.name,
        "mixed_scenario": mixed.name,
        "reference_train_rows": int(ref.n_training_rows()),
        "mixed_train_rows": int(mixed.n_training_rows()),
        "removed_rows": int(ref.n_training_rows() - mixed.n_training_rows()),
        "removed_fraction": float(1.0 - mixed.n_training_rows() / ref.n_training_rows()),
    }


K_RANDOM_SCENARIO_NAMES = {
    "FULL16_REFERENCE": "K_RANDOM_FULL16_REFERENCE",
    "MOD8": "K_RANDOM_RANK4_MOD8",
}


def _support_hamming_diversity(codes: Sequence[int]) -> int:
    """Sum of pairwise bitwise Hamming distances inside one support."""
    cc = tuple(map(int, codes))
    total = 0
    for a, b in itertools.combinations(cc, 2):
        total += int(np.sum(REGIME_BITS[a] != REGIME_BITS[b]))
    return int(total)


def select_maximally_diverse_rank4_size5_supports(n_supports: int = 8) -> Tuple[Tuple[int, ...], ...]:
    """Exact combinatorial selector for K-RANDOM's eight rank-4 supports.

    Every support is origin + four nonzero linearly independent vectors.  With
    only 15 nonzero vertices, 8 supports require 32 nonzero slots, so literal
    pairwise disjointness is impossible.  The lexicographic optimum enforced
    here is stronger than a random draw:

      1. each selected support has the globally maximal within-support Hamming
         diversity among all rank-4 size-5 supports;
      2. the eight supports jointly cover all 15 nonzero treatment vertices;
      3. any two supports share at most one nonzero vertex (the minimum possible
         maximum overlap once eight 4-vector bases are required);
      4. usage of the 15 nonzero vertices is maximally balanced: because there
         are 32 slots, the optimum frequency multiset is thirteen 2's and two
         3's.

    The candidate space is tiny for K=4: C(15,4)=1365 subsets, of which only
    full-rank subsets are retained.  A deterministic backtracking search then
    finds the first lexicographically ordered solution satisfying all optimality
    conditions.  No outcome information is used.
    """
    if K_BITS != 4 or N_REGIMES != 16 or int(n_supports) != 8:
        raise ValueError("K-RANDOM combinatorial selector is defined for K=4 and eight supports.")

    candidates: List[Tuple[int, ...]] = []
    scores: List[int] = []
    for basis in itertools.combinations(range(1, N_REGIMES), K_BITS):
        if rank_of_codes(basis) != K_BITS:
            continue
        support = (ORIGIN, *tuple(map(int, basis)))
        candidates.append(tuple(map(int, basis)))
        scores.append(_support_hamming_diversity(support))

    if not candidates:
        raise RuntimeError("K-RANDOM found no full-rank treatment bases.")
    max_diversity = int(max(scores))
    max_candidates = [
        basis for basis, score in zip(candidates, scores) if int(score) == max_diversity
    ]
    max_candidates = sorted(max_candidates)
    candidate_sets = [set(b) for b in max_candidates]

    target_min_freq = 2
    target_max_freq = 3
    # 8*4 = 32 slots.  With all 15 vertices covered at least twice, exactly two
    # vertices must occur a third time: 15*2 + 2 = 32.
    target_n_freq3 = n_supports * K_BITS - (N_REGIMES - 1) * target_min_freq
    chosen_solution: List[int] | None = None

    def backtrack(start: int, chosen: List[int], freq: List[int]) -> bool:
        nonlocal chosen_solution
        depth = len(chosen)
        if depth == n_supports:
            counts = [freq[v] for v in range(1, N_REGIMES)]
            if (
                min(counts) >= target_min_freq
                and max(counts) <= target_max_freq
                and sum(c == target_max_freq for c in counts) == target_n_freq3
            ):
                chosen_solution = list(chosen)
                return True
            return False

        remaining = n_supports - depth
        deficit = sum(max(0, target_min_freq - freq[v]) for v in range(1, N_REGIMES))
        if deficit > K_BITS * remaining:
            return False

        for idx in range(start, len(max_candidates)):
            s = candidate_sets[idx]
            if any(freq[v] >= target_max_freq for v in s):
                continue
            # Pairwise nonzero overlap <= 1.  Zero overlap for every pair is
            # impossible globally because 32 nonzero slots exceed 15 vertices.
            if any(len(s.intersection(candidate_sets[j])) > 1 for j in chosen):
                continue
            new_freq = list(freq)
            for v in s:
                new_freq[v] += 1
            if backtrack(idx + 1, chosen + [idx], new_freq):
                return True
        return False

    if not backtrack(0, [], [0] * N_REGIMES) or chosen_solution is None:
        raise RuntimeError("K-RANDOM could not solve the maximally diverse support design.")

    supports = tuple(
        tuple(sorted((ORIGIN, *max_candidates[idx]))) for idx in chosen_solution
    )

    # Strong proof-style guards for the claimed optimum properties.
    if len(supports) != n_supports or len(set(supports)) != n_supports:
        raise AssertionError("K-RANDOM supports are not eight distinct matrices.")
    if any(len(s) != 5 or ORIGIN not in s or rank_of_codes(s) != K_BITS for s in supports):
        raise AssertionError("K-RANDOM selected an invalid rank-4 size-5 support.")
    if any(_support_hamming_diversity(s) != max_diversity for s in supports):
        raise AssertionError("K-RANDOM selected a support below maximum within-matrix diversity.")

    nonzero_sets = [set(s) - {ORIGIN} for s in supports]
    union_nonzero = set().union(*nonzero_sets)
    if union_nonzero != set(range(1, N_REGIMES)):
        raise AssertionError("K-RANDOM supports do not cover all 15 nonzero vertices.")
    if max(len(a.intersection(b)) for a, b in itertools.combinations(nonzero_sets, 2)) > 1:
        raise AssertionError("K-RANDOM pairwise nonzero overlap exceeds one vertex.")
    freq = {v: sum(v in s for s in nonzero_sets) for v in range(1, N_REGIMES)}
    if sorted(freq.values()) != [2] * 13 + [3] * 2:
        raise AssertionError("K-RANDOM vertex usage is not maximally balanced.")

    return supports


def build_k_random_mod8_scenarios(
    data: DGPData,
    full2k_scenario: Scenario,
    region_labels: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, Scenario], Dict[str, object]]:
    """Construct K-RANDOM: eight interleaved, diverse, minimal rank-4 supports.

    The same literal full-2^K anchors are used in a matched full-support
    reference and in a mixed MOD8 design.  Inside every X region, anchors are
    ordered by S_extra and assigned by position modulo eight.  Each modulo class
    receives one of the eight combinatorially selected rank-4 size-5 supports.
    Assignment is seed-permuted to avoid tying a particular basis to one ordered
    position across Monte-Carlo replications.
    """
    if not full2k_scenario.has_training_override():
        raise ValueError("K-RANDOM requires the literal full-2^K|x baseline overrides.")

    full_support = tuple(range(N_REGIMES))
    base_supports = select_maximally_diverse_rank4_size5_supports(8)
    rng = np.random.default_rng(int(seed) + 91_337)
    support_perm = tuple(map(int, rng.permutation(8)))
    support_by_mod = {
        int(mod): tuple(base_supports[support_perm[mod]]) for mod in range(8)
    }

    # The selected family must collectively expose every treatment vertex.
    support_union = tuple(sorted(set().union(*(set(s) for s in support_by_mod.values()))))
    if support_union != full_support:
        raise AssertionError("K-RANDOM modulo supports do not collectively cover all 16 vertices.")

    base_anchors = np.asarray(full2k_scenario.train_idx, dtype=int)
    group_parts: Dict[int, List[np.ndarray]] = {m: [] for m in range(8)}
    selected_parts: List[np.ndarray] = []
    region_counts: Dict[str, Dict[str, int]] = {}

    descending = bool(int(seed) % 2)
    for region in X_REGIONS:
        anchors = base_anchors[region_labels[base_anchors] == region]
        if len(anchors) < 8:
            raise RuntimeError(f"K-RANDOM needs at least eight full-2^K anchors in {region}.")
        order = np.argsort(data.s_extra[anchors], kind="mergesort")
        ordered = anchors[order]
        if descending:
            ordered = ordered[::-1]

        n8 = 8 * (len(ordered) // 8)
        ordered = np.asarray(ordered[:n8], dtype=int)
        if len(ordered) < 8:
            raise RuntimeError(f"K-RANDOM has insufficient anchors in {region} after trimming.")
        selected_parts.append(ordered)

        counts: Dict[str, int] = {"n_selected": int(len(ordered))}
        for mod in range(8):
            grp = np.asarray(ordered[mod::8], dtype=int)
            group_parts[mod].append(grp)
            counts[f"n_mod{mod}"] = int(len(grp))
        if len({counts[f"n_mod{m}"] for m in range(8)}) != 1:
            raise AssertionError("K-RANDOM modulo groups must be equal within each X region.")
        region_counts[region] = counts

    group_idx = {m: np.concatenate(parts) for m, parts in group_parts.items()}
    selected_idx = np.concatenate(selected_parts)
    group_sizes = {m: int(len(idx)) for m, idx in group_idx.items()}
    if len(set(group_sizes.values())) != 1:
        raise AssertionError("K-RANDOM modulo groups must have equal global size.")

    stacked = np.concatenate([group_idx[m] for m in range(8)])
    if len(np.unique(stacked)) != len(stacked) or set(map(int, stacked)) != set(map(int, selected_idx)):
        raise AssertionError("K-RANDOM modulo partition is not exact.")

    # Nearby complementary-support diagnostics.  For each target X, find the
    # nearest anchor in the same physical region assigned to a *different*
    # support matrix.  This quantifies the local-information-sharing design.
    anchor_mod: Dict[int, int] = {}
    for mod, idx in group_idx.items():
        anchor_mod.update({int(x): int(mod) for x in idx})

    nearest_ds_parts: List[np.ndarray] = []
    nearest_dx_parts: List[np.ndarray] = []
    for region in X_REGIONS:
        region_selected = selected_idx[region_labels[selected_idx] == region]
        for mod in range(8):
            target = group_idx[mod][region_labels[group_idx[mod]] == region]
            other = np.asarray([x for x in region_selected if anchor_mod[int(x)] != mod], dtype=int)
            if not len(target) or not len(other):
                raise RuntimeError(f"K-RANDOM missing target/complementary anchors in {region}, mod={mod}.")
            ds_mat = np.abs(data.s_extra[target, None] - data.s_extra[other][None, :])
            nearest_ds_parts.append(np.min(ds_mat, axis=1))
            dmat = cdist(data.X_star[target], data.X_star[other], metric="euclidean")
            nearest_dx_parts.append(np.min(dmat, axis=1))
    nearest_ds = np.concatenate(nearest_ds_parts)
    nearest_dx = np.concatenate(nearest_dx_parts)

    base_src = full2k_scenario.training_source_idx()
    base_t = full2k_scenario.training_t(data)
    base_y_clean = np.asarray(full2k_scenario.train_y_clean_override, dtype=float)
    base_y_cont = np.asarray(full2k_scenario.train_y_contaminated_override, dtype=float)

    selected_mask = np.isin(base_src, selected_idx)
    ref_src = base_src[selected_mask]
    ref_t = base_t[selected_mask]
    ref_y_clean = base_y_clean[selected_mask]
    ref_y_cont = base_y_cont[selected_mask]

    keep_mixed = np.zeros(len(ref_src), dtype=bool)
    for i, (x, t) in enumerate(zip(ref_src, ref_t)):
        mod = anchor_mod[int(x)]
        keep_mixed[i] = int(t) in set(support_by_mod[mod])
    mix_src = ref_src[keep_mixed]
    mix_t = ref_t[keep_mixed]
    mix_y_clean = ref_y_clean[keep_mixed]
    mix_y_cont = ref_y_cont[keep_mixed]

    # Exact per-X guards.
    for mod in range(8):
        expected_mixed = tuple(sorted(support_by_mod[mod]))
        if len(expected_mixed) != 5 or rank_of_codes(expected_mixed) != K_BITS:
            raise AssertionError(f"K-RANDOM mod {mod} support is not rank4/size5.")
        for x in group_idx[mod]:
            ref_obs = tuple(sorted(np.unique(ref_t[ref_src == x]).astype(int).tolist()))
            mix_obs = tuple(sorted(np.unique(mix_t[mix_src == x]).astype(int).tolist()))
            if ref_obs != full_support:
                raise AssertionError(f"K-RANDOM reference lost treatment rows at x={int(x)}.")
            if mix_obs != expected_mixed:
                raise AssertionError(
                    f"K-RANDOM support mismatch at x={int(x)}, mod={mod}: "
                    f"got {bit_strings(mix_obs)}, expected {bit_strings(expected_mixed)}."
                )

    # Diversity diagnostics of the eight selected matrices.
    nonzero_sets = [set(support_by_mod[m]) - {ORIGIN} for m in range(8)]
    usage = {v: sum(v in s for s in nonzero_sets) for v in range(1, N_REGIMES)}
    overlaps = [len(a.intersection(b)) for a, b in itertools.combinations(nonzero_sets, 2)]
    internal_div = [_support_hamming_diversity(support_by_mod[m]) for m in range(8)]

    common_meta = {
        "evaluation_mode": "mod8_diverse_rank4_all16",
        "kr_selected_anchor_idx": tuple(map(int, selected_idx)),
        "kr_group_anchor_idx": {int(k): tuple(map(int, v)) for k, v in group_idx.items()},
        "kr_support_codes": {int(k): tuple(map(int, v)) for k, v in support_by_mod.items()},
        "kr_support_bits": {int(k): tuple(bit_string(c) for c in v) for k, v in support_by_mod.items()},
        "kr_support_permutation": support_perm,
        "kr_group_sizes": group_sizes,
        "kr_order_direction": "descending" if descending else "ascending",
        "kr_support_union_codes": support_union,
        "kr_support_union_bits": tuple(bit_string(c) for c in support_union),
        "kr_nonzero_usage": usage,
        "kr_max_pairwise_nonzero_overlap": int(max(overlaps)),
        "kr_mean_pairwise_nonzero_overlap": float(np.mean(overlaps)),
        "kr_internal_hamming_diversity": tuple(map(int, internal_div)),
        "kr_internal_hamming_diversity_min": int(min(internal_div)),
        "kr_neighbor_s_extra_mean": float(np.mean(nearest_ds)),
        "kr_neighbor_s_extra_median": float(np.median(nearest_ds)),
        "kr_neighbor_s_extra_p95": float(np.quantile(nearest_ds, 0.95)),
        "kr_neighbor_xstar_l2_mean": float(np.mean(nearest_dx)),
        "kr_neighbor_xstar_l2_median": float(np.median(nearest_dx)),
        "kr_neighbor_xstar_l2_p95": float(np.quantile(nearest_dx, 0.95)),
        "kr_neighbor_definition": "nearest anchor with a different rank-4 support within same X region",
        "kr_region_counts": region_counts,
    }

    ref = Scenario(
        name=K_RANDOM_SCENARIO_NAMES["FULL16_REFERENCE"],
        train_idx=np.asarray(selected_idx, dtype=int),
        test_idx=np.asarray(selected_idx, dtype=int),
        description=(
            "K-RANDOM matched reference: the same modulo-8 X anchors are used, "
            "but every selected anchor retains all 16 treatment vertices."
        ),
        local_supports={r: full_support for r in X_REGIONS},
        local_bases={r: tuple(CANONICAL_NONZERO) for r in X_REGIONS},
        target_kind="neighbor_mod8_random_rank4_x",
        family="K_RANDOM_MOD8_RANK4_EXTRAPOLATION",
        metadata={**common_meta, "kr_profile": "FULL16_REFERENCE", "literal_all_2k_per_x": 1},
        train_source_idx_override=np.asarray(ref_src, dtype=int),
        train_t_override=np.asarray(ref_t, dtype=int),
        train_y_clean_override=np.asarray(ref_y_clean, dtype=float),
        train_y_contaminated_override=np.asarray(ref_y_cont, dtype=float),
    )

    mixed = Scenario(
        name=K_RANDOM_SCENARIO_NAMES["MOD8"],
        train_idx=np.asarray(selected_idx, dtype=int),
        test_idx=np.asarray(selected_idx, dtype=int),
        description=(
            "K-RANDOM modulo-8 extrapolation: every X anchor has a minimal rank-4 "
            "five-vertex support, but eight maximally diverse bases are interleaved "
            "throughout each X region; all 16 potential outcomes are evaluated."
        ),
        # Region-level union is full16 because the eight local bases jointly cover all vertices.
        local_supports={r: full_support for r in X_REGIONS},
        local_bases={r: tuple(CANONICAL_NONZERO) for r in X_REGIONS},
        target_kind="neighbor_mod8_random_rank4_x",
        family="K_RANDOM_MOD8_RANK4_EXTRAPOLATION",
        metadata={**common_meta, "kr_profile": "MOD8", "literal_all_2k_per_x": 0},
        train_source_idx_override=np.asarray(mix_src, dtype=int),
        train_t_override=np.asarray(mix_t, dtype=int),
        train_y_clean_override=np.asarray(mix_y_clean, dtype=float),
        train_y_contaminated_override=np.asarray(mix_y_cont, dtype=float),
    )

    return {ref.name: ref, mixed.name: mixed}, {
        **common_meta,
        "reference_scenario": ref.name,
        "mixed_scenario": mixed.name,
        "reference_train_rows": int(ref.n_training_rows()),
        "mixed_train_rows": int(mixed.n_training_rows()),
        "removed_rows": int(ref.n_training_rows() - mixed.n_training_rows()),
        "removed_fraction": float(1.0 - mixed.n_training_rows() / ref.n_training_rows()),
    }


K_RANDOM3_SCENARIO_NAMES = {
    "MOD8": "K_RANDOM3_RANK32_MOD8",
}


def select_maximally_diverse_rank3_rank2_mod8_supports() -> Tuple[
    Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]
]:
    """Return the fixed optimal K-RANDOM3 support family for K=4.

    Four supports have rank 3 and total size 4 (origin + 3 nonzero vectors),
    while four supports have rank 2 and total size 3 (origin + 2 nonzero
    vectors).  The eight supports jointly cover every one of the 15 nonzero
    treatment vertices, but no individual support has rank 4.

    The selected family is the deterministic solution of the finite K=4
    combinatorial design problem with the following lexicographic goals:

      1. collective coverage of all 15 nonzero treatment vertices;
      2. maximum total within-support pairwise Hamming diversity;
      3. maximum nonzero-vertex usage balance;
      4. pairwise nonzero overlap at most one vertex.

    With four rank-3 supports and four rank-2 supports there are 20 nonzero
    slots.  Full coverage therefore has an ideal frequency multiset of ten 1's
    and five 2's, which the design attains.  The unconstrained Hamming upper
    bound is 4*16 + 4*8 = 96; exhaustive enumeration shows that no score-96
    selection covers all 15 nonzero vertices, while the design below scores 95.
    Hence 95 is optimal under the full-coverage constraint.
    """
    if K_BITS != 4 or N_REGIMES != 16:
        raise ValueError("K-RANDOM3 support selector is defined for K=4.")

    # Precomputed exact optimum for the tiny K=4 combinatorial problem.
    rank3_nonzero = (
        (4, 9, 15),
        (5, 11, 14),
        (6, 11, 13),
        (7, 10, 13),
    )
    rank2_nonzero = (
        (1, 14),
        (2, 15),
        (3, 12),
        (7, 8),
    )
    rank3_supports = tuple(tuple(sorted((ORIGIN, *v))) for v in rank3_nonzero)
    rank2_supports = tuple(tuple(sorted((ORIGIN, *v))) for v in rank2_nonzero)
    all_supports = rank3_supports + rank2_supports

    if any(len(s) != 4 or rank_of_codes(s) != 3 for s in rank3_supports):
        raise AssertionError("K-RANDOM3 contains an invalid rank-3 support.")
    if any(len(s) != 3 or rank_of_codes(s) != 2 for s in rank2_supports):
        raise AssertionError("K-RANDOM3 contains an invalid rank-2 support.")
    if any(rank_of_codes(s) == K_BITS for s in all_supports):
        raise AssertionError("K-RANDOM3 must never expose a local rank-4 support.")

    nonzero_sets = [set(s) - {ORIGIN} for s in all_supports]
    union_nonzero = set().union(*nonzero_sets)
    if union_nonzero != set(range(1, N_REGIMES)):
        raise AssertionError("K-RANDOM3 supports do not cover all 15 nonzero vertices.")

    freq = {v: sum(v in s for s in nonzero_sets) for v in range(1, N_REGIMES)}
    if sorted(freq.values()) != [1] * 10 + [2] * 5:
        raise AssertionError("K-RANDOM3 nonzero-vertex usage is not maximally balanced.")

    overlaps = [len(a.intersection(b)) for a, b in itertools.combinations(nonzero_sets, 2)]
    if max(overlaps) > 1:
        raise AssertionError("K-RANDOM3 pairwise nonzero overlap exceeds one vertex.")

    diversity = [_support_hamming_diversity(s) for s in all_supports]
    if sum(diversity) != 95:
        raise AssertionError("K-RANDOM3 support family lost its optimal Hamming diversity score.")
    if sorted(_support_hamming_diversity(s) for s in rank2_supports) != [8, 8, 8, 8]:
        raise AssertionError("K-RANDOM3 rank-2 supports are not individually maximally diverse.")
    if sorted(_support_hamming_diversity(s) for s in rank3_supports) != [15, 16, 16, 16]:
        raise AssertionError("K-RANDOM3 rank-3 diversity profile changed unexpectedly.")

    return rank3_supports, rank2_supports


def build_k_random3_mod8_scenario(
    data: DGPData,
    k_random_scenarios: Mapping[str, Scenario],
    k_random_meta: Mapping[str, object],
    seed: int,
) -> Tuple[Dict[str, Scenario], Dict[str, object]]:
    """Construct K-RANDOM3 on the exact same ordered-X anchors as K-RANDOM4.

    Even modulo classes receive rank-3 supports and odd modulo classes receive
    rank-2 supports.  All eight supports are distinct and their union is full16,
    but no individual training X has rank 4.  The existing K-RANDOM full16
    reference is reused, and K-RANDOM3-specific evaluation metadata is attached
    to that reference so matched contrasts require only one additional model fit.
    """
    ref_name = K_RANDOM_SCENARIO_NAMES["FULL16_REFERENCE"]
    if ref_name not in k_random_scenarios:
        raise ValueError("K-RANDOM3 requires the K-RANDOM full16 reference scenario.")
    ref = k_random_scenarios[ref_name]
    if not ref.has_training_override():
        raise ValueError("K-RANDOM3 requires a literal full16 training override reference.")

    selected_idx = np.asarray(k_random_meta["kr_selected_anchor_idx"], dtype=int)
    group_idx = {
        int(k): np.asarray(v, dtype=int)
        for k, v in k_random_meta["kr_group_anchor_idx"].items()
    }
    if set(group_idx) != set(range(8)):
        raise AssertionError("K-RANDOM3 expected exactly eight modulo groups.")
    stacked = np.concatenate([group_idx[m] for m in range(8)])
    if len(np.unique(stacked)) != len(stacked) or set(map(int, stacked)) != set(map(int, selected_idx)):
        raise AssertionError("K-RANDOM3 did not inherit an exact modulo-8 anchor partition.")

    rank3_base, rank2_base = select_maximally_diverse_rank3_rank2_mod8_supports()
    rng = np.random.default_rng(int(seed) + 193_771)
    perm3 = tuple(map(int, rng.permutation(4)))
    perm2 = tuple(map(int, rng.permutation(4)))
    even_mods = (0, 2, 4, 6)
    odd_mods = (1, 3, 5, 7)
    support_by_mod: Dict[int, Tuple[int, ...]] = {}
    for j, mod in enumerate(even_mods):
        support_by_mod[mod] = tuple(rank3_base[perm3[j]])
    for j, mod in enumerate(odd_mods):
        support_by_mod[mod] = tuple(rank2_base[perm2[j]])
    rank_by_mod = {m: int(rank_of_codes(support_by_mod[m])) for m in range(8)}
    cardinality_by_mod = {m: int(len(support_by_mod[m])) for m in range(8)}

    if any(rank_by_mod[m] != 3 or cardinality_by_mod[m] != 4 for m in even_mods):
        raise AssertionError("K-RANDOM3 even modulo classes must be rank3/size4.")
    if any(rank_by_mod[m] != 2 or cardinality_by_mod[m] != 3 for m in odd_mods):
        raise AssertionError("K-RANDOM3 odd modulo classes must be rank2/size3.")
    if any(rank_by_mod[m] >= K_BITS for m in range(8)):
        raise AssertionError("K-RANDOM3 accidentally exposed rank 4 at a local X support.")

    full_support = tuple(range(N_REGIMES))
    support_union = tuple(sorted(set().union(*(set(s) for s in support_by_mod.values()))))
    if support_union != full_support:
        raise AssertionError("K-RANDOM3 modulo supports do not collectively cover full16.")

    ref_src = ref.training_source_idx()
    ref_t = ref.training_t(data)
    ref_y_clean = np.asarray(ref.train_y_clean_override, dtype=float)
    ref_y_cont = np.asarray(ref.train_y_contaminated_override, dtype=float)
    if set(map(int, np.unique(ref_src))) != set(map(int, selected_idx)):
        raise AssertionError("K-RANDOM3 shared reference anchors differ from K-RANDOM anchors.")

    anchor_mod: Dict[int, int] = {}
    for mod, idx in group_idx.items():
        anchor_mod.update({int(x): int(mod) for x in idx})

    keep_mixed = np.zeros(len(ref_src), dtype=bool)
    support_sets = {m: set(support_by_mod[m]) for m in range(8)}
    for i, (x, t) in enumerate(zip(ref_src, ref_t)):
        keep_mixed[i] = int(t) in support_sets[anchor_mod[int(x)]]
    mix_src = ref_src[keep_mixed]
    mix_t = ref_t[keep_mixed]
    mix_y_clean = ref_y_clean[keep_mixed]
    mix_y_cont = ref_y_cont[keep_mixed]

    for mod in range(8):
        expected = tuple(sorted(support_by_mod[mod]))
        for x in group_idx[mod]:
            ref_obs = tuple(sorted(np.unique(ref_t[ref_src == x]).astype(int).tolist()))
            mix_obs = tuple(sorted(np.unique(mix_t[mix_src == x]).astype(int).tolist()))
            if ref_obs != full_support:
                raise AssertionError(f"K-RANDOM3 reference lost treatment rows at x={int(x)}.")
            if mix_obs != expected:
                raise AssertionError(
                    f"K-RANDOM3 support mismatch at x={int(x)}, mod={mod}: "
                    f"got {bit_strings(mix_obs)}, expected {bit_strings(expected)}."
                )
            if rank_of_codes(mix_obs) not in (2, 3):
                raise AssertionError("K-RANDOM3 mixed support must never have rank 4.")

    nonzero_sets = [set(support_by_mod[m]) - {ORIGIN} for m in range(8)]
    usage = {v: sum(v in ss for ss in nonzero_sets) for v in range(1, N_REGIMES)}
    overlaps = [len(a.intersection(b)) for a, b in itertools.combinations(nonzero_sets, 2)]
    internal_div = [_support_hamming_diversity(support_by_mod[m]) for m in range(8)]
    group_sizes = {m: int(len(group_idx[m])) for m in range(8)}

    common_meta = {
        "kr3_evaluation_mode": "mod8_interleaved_rank3_rank2_all16",
        "kr3_selected_anchor_idx": tuple(map(int, selected_idx)),
        "kr3_group_anchor_idx": {int(k): tuple(map(int, v)) for k, v in group_idx.items()},
        "kr3_support_codes": {int(k): tuple(map(int, v)) for k, v in support_by_mod.items()},
        "kr3_support_bits": {int(k): tuple(bit_string(c) for c in v) for k, v in support_by_mod.items()},
        "kr3_rank_by_mod": rank_by_mod,
        "kr3_cardinality_by_mod": cardinality_by_mod,
        "kr3_rank3_support_permutation": perm3,
        "kr3_rank2_support_permutation": perm2,
        "kr3_group_sizes": group_sizes,
        "kr3_order_direction": str(k_random_meta["kr_order_direction"]),
        "kr3_support_union_codes": support_union,
        "kr3_support_union_bits": tuple(bit_string(c) for c in support_union),
        "kr3_nonzero_usage": usage,
        "kr3_max_pairwise_nonzero_overlap": int(max(overlaps)),
        "kr3_mean_pairwise_nonzero_overlap": float(np.mean(overlaps)),
        "kr3_internal_hamming_diversity": tuple(map(int, internal_div)),
        "kr3_total_hamming_diversity": int(sum(internal_div)),
        "kr3_neighbor_s_extra_mean": float(k_random_meta["kr_neighbor_s_extra_mean"]),
        "kr3_neighbor_s_extra_median": float(k_random_meta["kr_neighbor_s_extra_median"]),
        "kr3_neighbor_s_extra_p95": float(k_random_meta["kr_neighbor_s_extra_p95"]),
        "kr3_neighbor_xstar_l2_mean": float(k_random_meta["kr_neighbor_xstar_l2_mean"]),
        "kr3_neighbor_xstar_l2_median": float(k_random_meta["kr_neighbor_xstar_l2_median"]),
        "kr3_neighbor_xstar_l2_p95": float(k_random_meta["kr_neighbor_xstar_l2_p95"]),
        "kr3_neighbor_definition": "same K-RANDOM modulo-8 X partition; nearest different support within region",
    }

    # Attach the conceptual K-RANDOM3 support partition to the already-existing
    # full16 reference.  Its fit is reused; evaluate_one_scenario will emit both
    # kr_* and kr3_* matched metrics for this one reference row.
    ref.metadata.update({**common_meta, "kr3_profile": "FULL16_REFERENCE"})

    mixed = Scenario(
        name=K_RANDOM3_SCENARIO_NAMES["MOD8"],
        train_idx=np.asarray(selected_idx, dtype=int),
        test_idx=np.asarray(selected_idx, dtype=int),
        description=(
            "K-RANDOM3 modulo-8 extrapolation: the same ordered-X partition as K-RANDOM is used, "
            "but local training supports alternate rank3/size4 and rank2/size3; no X has local "
            "rank4 support, the eight supports collectively cover all 16 treatment vertices, "
            "and all 16 potential outcomes are evaluated."
        ),
        local_supports={r: full_support for r in X_REGIONS},
        local_bases={r: tuple(CANONICAL_NONZERO) for r in X_REGIONS},
        target_kind="neighbor_mod8_random_rank32_x",
        family="K_RANDOM3_MOD8_RANK32_EXTRAPOLATION",
        metadata={**common_meta, "kr3_profile": "MOD8", "literal_all_2k_per_x": 0},
        train_source_idx_override=np.asarray(mix_src, dtype=int),
        train_t_override=np.asarray(mix_t, dtype=int),
        train_y_clean_override=np.asarray(mix_y_clean, dtype=float),
        train_y_contaminated_override=np.asarray(mix_y_cont, dtype=float),
    )

    return {mixed.name: mixed}, {
        **common_meta,
        "reference_scenario": ref.name,
        "mixed_scenario": mixed.name,
        "reference_train_rows": int(ref.n_training_rows()),
        "mixed_train_rows": int(mixed.n_training_rows()),
        "removed_rows": int(ref.n_training_rows() - mixed.n_training_rows()),
        "removed_fraction": float(1.0 - mixed.n_training_rows() / ref.n_training_rows()),
    }




K_EXTREME_SCENARIO_NAMES = {
    "RANK4": "K_EXTREME_RANK4_MAXMOD",
    "LT4": "K_EXTREME_LT4_MAXMOD",
}


def enumerate_all_minimal_supports_of_rank(rank: int) -> Tuple[Tuple[int, ...], ...]:
    """Enumerate every minimal origin-containing support of an exact rank.

    A minimal rank-r support is ``{0,u1,...,ur}`` with r independent nonzero
    treatment vectors.  Restricting to minimal supports keeps the extreme test
    focused on *orientation diversity* rather than mixing in arbitrary redundant
    cardinalities.
    """
    rank = int(rank)
    if K_BITS != 4 or N_REGIMES != 16:
        raise ValueError("K-EXTREME exact catalogs are defined for K=4.")
    if rank < 1 or rank > K_BITS:
        raise ValueError(f"rank must be in 1..{K_BITS}; received {rank}.")

    supports: List[Tuple[int, ...]] = []
    for basis in itertools.combinations(range(1, N_REGIMES), rank):
        if rank_of_codes(basis) != rank:
            continue
        supports.append((ORIGIN, *tuple(map(int, basis))))
    supports = sorted(set(supports))

    expected = {1: 15, 2: 105, 3: 430, 4: 940}[rank]
    if len(supports) != expected:
        raise AssertionError(
            f"K-EXTREME rank-{rank} catalog has {len(supports)} supports; expected {expected}."
        )
    if any(len(s) != rank + 1 or rank_of_codes(s) != rank for s in supports):
        raise AssertionError(f"K-EXTREME rank-{rank} catalog contains invalid supports.")
    return tuple(supports)


def _extreme_catalogs() -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
    """Return the complete minimal rank-4 and rank<4 support catalogs."""
    rank4 = enumerate_all_minimal_supports_of_rank(4)
    rank1 = enumerate_all_minimal_supports_of_rank(1)
    rank2 = enumerate_all_minimal_supports_of_rank(2)
    rank3 = enumerate_all_minimal_supports_of_rank(3)
    lt4 = tuple((*rank1, *rank2, *rank3))
    if len(rank4) != 940 or len(lt4) != 550:
        raise AssertionError("K-EXTREME theoretical module sizes changed unexpectedly.")
    if tuple(sorted(set().union(*(set(s) for s in rank4)))) != tuple(range(N_REGIMES)):
        raise AssertionError("K-EXTREME rank4 catalog does not cover full16.")
    if tuple(sorted(set().union(*(set(s) for s in lt4)))) != tuple(range(N_REGIMES)):
        raise AssertionError("K-EXTREME rank<4 catalog does not cover full16.")
    if any(rank_of_codes(s) >= K_BITS for s in lt4):
        raise AssertionError("K-EXTREME rank<4 catalog accidentally contains rank4 support.")
    return rank4, lt4


def _assign_extreme_catalog_to_ordered_anchors(
    ordered_anchor_idx: np.ndarray,
    catalog: Sequence[Sequence[int]],
    seed: int,
) -> Tuple[Dict[int, Tuple[int, ...]], Tuple[Tuple[int, ...], ...], int]:
    """Assign as many distinct supports as possible before cycling.

    The anchor order is inherited from K-RANDOM: X is ordered within each Q
    region and the region-wise ordered arrays are concatenated.  The complete
    catalog is seed-permuted without replacement.  Hence the first cycle uses
    every feasible support exactly once, maximizing the modulo size subject to
    the available number of X anchors.
    """
    anchors = np.asarray(ordered_anchor_idx, dtype=int)
    if len(anchors) == 0:
        raise ValueError("K-EXTREME received no anchors.")
    base = tuple(tuple(map(int, s)) for s in catalog)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(base))
    permuted = tuple(base[int(j)] for j in perm)
    effective_module = int(min(len(permuted), len(anchors)))
    active = permuted[:effective_module]
    if len(set(active)) != effective_module:
        raise AssertionError("K-EXTREME active catalog is not unique within its first cycle.")

    assigned = {
        int(x): tuple(active[pos % effective_module])
        for pos, x in enumerate(anchors)
    }
    if len(set(assigned.values())) != effective_module:
        raise AssertionError("K-EXTREME failed to realize its maximum feasible module.")
    return assigned, active, effective_module


def _filter_full16_reference_by_anchor_support(
    ref: Scenario,
    data: DGPData,
    support_by_anchor: Mapping[int, Sequence[int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Delete only treatment rows not present in each anchor-specific support."""
    src = ref.training_source_idx()
    tt = ref.training_t(data)
    y_clean = np.asarray(ref.train_y_clean_override, dtype=float)
    y_cont = np.asarray(ref.train_y_contaminated_override, dtype=float)
    support_sets = {int(x): set(map(int, s)) for x, s in support_by_anchor.items()}
    keep = np.fromiter(
        (int(t) in support_sets[int(x)] for x, t in zip(src, tt)),
        dtype=bool,
        count=len(src),
    )
    return src[keep], tt[keep], y_clean[keep], y_cont[keep]


def build_k_extreme_scenarios(
    data: DGPData,
    k_random_scenarios: Mapping[str, Scenario],
    k_random_meta: Mapping[str, object],
    seed: int,
) -> Tuple[Dict[str, Scenario], Dict[str, object]]:
    """Build maximum-module K-RANDOM variants on the same ordered-X anchors.

    Two mixed designs are created and the existing full16 K-RANDOM reference is
    reused:

      * RANK4: every possible minimal rank-4 support (940 theoretical variants);
      * LT4: every possible minimal rank-1/2/3 support (550 variants total).

    When the sample provides fewer anchors than a catalog contains, the first
    cycle uses one distinct support per anchor, which is the largest feasible
    empirical module.  No outcome information enters support construction.
    """
    ref_name = K_RANDOM_SCENARIO_NAMES["FULL16_REFERENCE"]
    if ref_name not in k_random_scenarios:
        raise ValueError("K-EXTREME requires the K-RANDOM full16 reference.")
    ref = k_random_scenarios[ref_name]
    if not ref.has_training_override():
        raise ValueError("K-EXTREME requires a literal full16 override reference.")

    selected_idx = np.asarray(k_random_meta["kr_selected_anchor_idx"], dtype=int)
    if len(selected_idx) != len(np.unique(selected_idx)):
        raise AssertionError("K-EXTREME requires unique ordered X anchors.")
    if set(map(int, np.unique(ref.training_source_idx()))) != set(map(int, selected_idx)):
        raise AssertionError("K-EXTREME reference anchors differ from K-RANDOM anchors.")

    rank4_catalog, lt4_catalog = _extreme_catalogs()
    r4_by_x, r4_active, r4_module = _assign_extreme_catalog_to_ordered_anchors(
        selected_idx, rank4_catalog, int(seed) + 401
    )
    lt4_by_x, lt4_active, lt4_module = _assign_extreme_catalog_to_ordered_anchors(
        selected_idx, lt4_catalog, int(seed) + 809
    )

    full_support = tuple(range(N_REGIMES))
    if tuple(sorted(set().union(*(set(v) for v in r4_by_x.values())))) != full_support:
        raise AssertionError("K-EXTREME rank4 empirical support union is not full16.")
    if tuple(sorted(set().union(*(set(v) for v in lt4_by_x.values())))) != full_support:
        raise AssertionError("K-EXTREME rank<4 empirical support union is not full16.")

    r4_src, r4_t, r4_y_clean, r4_y_cont = _filter_full16_reference_by_anchor_support(
        ref, data, r4_by_x
    )
    lt_src, lt_t, lt_y_clean, lt_y_cont = _filter_full16_reference_by_anchor_support(
        ref, data, lt4_by_x
    )

    # Fast exact per-X geometry guards.  The full reference has exactly one row
    # for each of the 16 treatments at every selected anchor.
    def observed_supports(src: np.ndarray, tt: np.ndarray) -> Dict[int, Tuple[int, ...]]:
        out: Dict[int, List[int]] = {}
        for x, t in zip(src, tt):
            out.setdefault(int(x), []).append(int(t))
        return {x: tuple(sorted(set(v))) for x, v in out.items()}

    r4_obs = observed_supports(r4_src, r4_t)
    lt_obs = observed_supports(lt_src, lt_t)
    for x in selected_idx:
        xx = int(x)
        exp4 = tuple(sorted(r4_by_x[xx]))
        explt = tuple(sorted(lt4_by_x[xx]))
        if r4_obs.get(xx, tuple()) != exp4:
            raise AssertionError(f"K-EXTREME rank4 support mismatch at x={xx}.")
        if lt_obs.get(xx, tuple()) != explt:
            raise AssertionError(f"K-EXTREME rank<4 support mismatch at x={xx}.")
        if rank_of_codes(exp4) != 4 or len(exp4) != 5:
            raise AssertionError("K-EXTREME rank4 anchor is not minimal rank4/size5.")
        rr = rank_of_codes(explt)
        if rr not in (1, 2, 3) or len(explt) != rr + 1:
            raise AssertionError("K-EXTREME rank<4 anchor is not a minimal rank1/2/3 support.")

    r4_rank_counts = {4: int(len(selected_idx))}
    lt_rank_counts = {
        r: int(sum(rank_of_codes(lt4_by_x[int(x)]) == r for x in selected_idx))
        for r in (1, 2, 3)
    }
    lt_card_counts = {
        r + 1: int(sum(len(lt4_by_x[int(x)]) == r + 1 for x in selected_idx))
        for r in (1, 2, 3)
    }

    common = {
        "kext_selected_anchor_idx": tuple(map(int, selected_idx)),
        "kext_order_direction": str(k_random_meta["kr_order_direction"]),
        "kext_reference_scenario": ref.name,
        "kext_reference_train_rows": int(ref.n_training_rows()),
        "kext_evaluation_mode": "maximum_module_neighbor_support_all16",
    }

    r4_meta = {
        **common,
        "kext_variant": "RANK4_MAXMOD",
        "kext_theoretical_module": int(len(rank4_catalog)),
        "kext_effective_module": int(r4_module),
        "kext_n_unique_supports_used": int(len(set(r4_by_x.values()))),
        "kext_anchor_support_codes": {int(x): tuple(map(int, s)) for x, s in r4_by_x.items()},
        "kext_rank_counts": r4_rank_counts,
        "kext_cardinality_counts": {5: int(len(selected_idx))},
        "kext_catalog_rank_counts": {4: int(len(rank4_catalog))},
        "kext_support_union_codes": full_support,
        "kext_active_catalog_hamming_mean": float(np.mean([_support_hamming_diversity(s) for s in r4_active])),
    }
    lt_meta = {
        **common,
        "kext_variant": "LT4_MAXMOD",
        "kext_theoretical_module": int(len(lt4_catalog)),
        "kext_effective_module": int(lt4_module),
        "kext_n_unique_supports_used": int(len(set(lt4_by_x.values()))),
        "kext_anchor_support_codes": {int(x): tuple(map(int, s)) for x, s in lt4_by_x.items()},
        "kext_rank_counts": lt_rank_counts,
        "kext_cardinality_counts": lt_card_counts,
        "kext_catalog_rank_counts": {1: 15, 2: 105, 3: 430},
        "kext_support_union_codes": full_support,
        "kext_active_catalog_hamming_mean": float(np.mean([_support_hamming_diversity(s) for s in lt4_active])),
    }

    r4 = Scenario(
        name=K_EXTREME_SCENARIO_NAMES["RANK4"],
        train_idx=np.asarray(selected_idx, dtype=int),
        test_idx=np.asarray(selected_idx, dtype=int),
        description=(
            "K-EXTREME rank4: maximum feasible module drawn without replacement from all "
            "940 minimal rank-4 size-5 supports, assigned along the same ordered-X anchors "
            "as K-RANDOM; all 16 potential outcomes are evaluated."
        ),
        local_supports={r: full_support for r in X_REGIONS},
        local_bases={r: tuple(CANONICAL_NONZERO) for r in X_REGIONS},
        target_kind="neighbor_maxmod_rank4_x",
        family="K_EXTREME_MAXMOD_EXTRAPOLATION",
        metadata={**r4_meta, "literal_all_2k_per_x": 0},
        train_source_idx_override=np.asarray(r4_src, dtype=int),
        train_t_override=np.asarray(r4_t, dtype=int),
        train_y_clean_override=np.asarray(r4_y_clean, dtype=float),
        train_y_contaminated_override=np.asarray(r4_y_cont, dtype=float),
    )
    lt4 = Scenario(
        name=K_EXTREME_SCENARIO_NAMES["LT4"],
        train_idx=np.asarray(selected_idx, dtype=int),
        test_idx=np.asarray(selected_idx, dtype=int),
        description=(
            "K-EXTREME rank<4: maximum feasible module drawn without replacement from all "
            "550 minimal rank-1/rank-2/rank-3 supports, assigned along the same ordered-X "
            "anchors as K-RANDOM; no local X has rank4 and all 16 outcomes are evaluated."
        ),
        local_supports={r: full_support for r in X_REGIONS},
        local_bases={r: tuple(CANONICAL_NONZERO) for r in X_REGIONS},
        target_kind="neighbor_maxmod_ranklt4_x",
        family="K_EXTREME_MAXMOD_EXTRAPOLATION",
        metadata={**lt_meta, "literal_all_2k_per_x": 0},
        train_source_idx_override=np.asarray(lt_src, dtype=int),
        train_t_override=np.asarray(lt_t, dtype=int),
        train_y_clean_override=np.asarray(lt_y_clean, dtype=float),
        train_y_contaminated_override=np.asarray(lt_y_cont, dtype=float),
    )

    # The full16 reference is already fitted for K-RANDOM.  Mark it so its
    # all-16 metrics can serve as the exact K-EXTREME matched reference too.
    ref.metadata["kext_reference_shared"] = 1

    return {r4.name: r4, lt4.name: lt4}, {
        "reference_scenario": ref.name,
        "rank4_scenario": r4.name,
        "lt4_scenario": lt4.name,
        "rank4_theoretical_module": 940,
        "rank4_effective_module": int(r4_module),
        "lt4_theoretical_module": 550,
        "lt4_effective_module": int(lt4_module),
        "rank4_train_rows": int(r4.n_training_rows()),
        "lt4_train_rows": int(lt4.n_training_rows()),
    }


I_THEOREM_SAFE_ORDER = (
    "FULL16",
    "R4_REDUNDANT",
    "R4_CANONICAL",
    "R3_REDUNDANT",
)

I_THEOREM_SAFE_SCENARIO_NAMES = {
    "FULL16": "I_THEOREM_SAFE_FULL16",
    "R4_REDUNDANT": "I_THEOREM_SAFE_R4_REDUNDANT",
    "R4_CANONICAL": "I_THEOREM_SAFE_R4_CANONICAL",
    "R3_REDUNDANT": "I_THEOREM_SAFE_R3_REDUNDANT_SUPPORTED_ONLY",
}


def choose_i_global_redundant_vertex(
    data: DGPData,
    train_pool: np.ndarray,
    region_labels: np.ndarray,
    retained_basis: Sequence[int],
    rng: np.random.Generator,
) -> int:
    """Choose one redundant vertex supported in *every* X region.

    The selected noncanonical vertex v must lie in span(retained_basis), so
    adding v does not restore the omitted fourth direction.  Selection uses
    treatment counts only (never outcomes) and favors vertices with a large
    minimum count across Q1..Q4.  Using the same v in all four regions makes the
    I scenarios geometrically identical across X and removes an X-extrapolation
    interpretation from the experiment.
    """
    retained_basis = tuple(map(int, retained_basis))
    if rank_of_codes(retained_basis) != K_BITS - 1:
        raise ValueError("I retained_basis must have rank K-1.")

    canonical = set(map(int, CANONICAL_NONZERO))
    region_counts: Dict[str, np.ndarray] = {}
    for region in X_REGIONS:
        ids = train_pool[region_labels[train_pool] == region]
        region_counts[region] = np.bincount(data.T[ids], minlength=N_REGIMES)

    candidates: List[Tuple[int, int, int]] = []
    for code in range(1, N_REGIMES):
        if code in canonical:
            continue
        if rank_of_codes((*retained_basis, int(code))) != K_BITS - 1:
            continue
        counts = [int(region_counts[r][code]) for r in X_REGIONS]
        if min(counts) <= 0:
            continue
        candidates.append((min(counts), sum(counts), int(code)))

    if not candidates:
        raise RuntimeError(
            "No noncanonical redundant vertex is empirically represented in all "
            "four X regions while remaining inside the retained rank-3 span."
        )

    best_min = max(x[0] for x in candidates)
    eligible = [x for x in candidates if x[0] >= 0.75 * best_min]
    best_total = max(x[1] for x in eligible)
    eligible = [x for x in eligible if x[1] >= 0.85 * best_total]
    eligible.sort(key=lambda z: (z[0], z[1], z[2]), reverse=True)
    top = eligible[: min(12, len(eligible))]
    return int(top[int(rng.integers(0, len(top)))][2])


def build_theorem_safe_support_ladder_scenarios(
    data: DGPData,
    train_pool: np.ndarray,
    population_eval_idx: np.ndarray,
    region_labels: np.ndarray,
    seed: int,
    max_train_n: int | None,
    cycle_index: int = 0,
) -> Tuple[Dict[str, Scenario], Dict[str, object]]:
    """Construct the I family: supported-only, no X extrapolation.

    Every scenario uses all four physical X strata with equal 25% training
    shares.  Within a given scenario, exactly the same treatment support is
    imposed in Q1, Q2, Q3 and Q4.  The evaluation set is the same representative
    population sample used elsewhere, but the treatment matrix supplied to the
    metrics is restricted to the scenario's supported actions.

    The common comparison core is {0,R}, where R contains the three canonical
    directions retained after one counterbalanced direction e_m is designated
    as the omitted direction.  This common core is observed in every X region in
    all four I profiles, so the cross-profile comparison never uses an
    unsupported treatment action.
    """
    full_support = tuple(range(N_REGIMES))
    full_basis = tuple(map(int, CANONICAL_NONZERO))

    missing_component = int(cycle_index) % K_BITS
    omitted_direction = int(CANONICAL_NONZERO[missing_component])
    retained_basis = tuple(
        int(code) for j, code in enumerate(CANONICAL_NONZERO)
        if j != missing_component
    )
    redundant_vertex = choose_i_global_redundant_vertex(
        data,
        train_pool,
        region_labels,
        retained_basis,
        np.random.default_rng(seed + 701),
    )

    r4_red_support = tuple(sorted((ORIGIN, *full_basis, redundant_vertex)))
    r4_can_support = tuple(sorted((ORIGIN, *full_basis)))
    r3_red_support = tuple(sorted((ORIGIN, *retained_basis, redundant_vertex)))
    common_core = tuple(sorted((ORIGIN, *retained_basis)))

    # Geometry and nesting guards.  R3 is intentionally a boundary condition;
    # it is evaluated only on its observed support/common core.
    if rank_of_codes(r4_red_support) != K_BITS or len(r4_red_support) != 6:
        raise AssertionError("I R4_REDUNDANT must have rank 4 and 6 vertices.")
    if rank_of_codes(r4_can_support) != K_BITS or len(r4_can_support) != 5:
        raise AssertionError("I R4_CANONICAL must have rank 4 and 5 vertices.")
    if rank_of_codes(r3_red_support) != K_BITS - 1 or len(r3_red_support) != 5:
        raise AssertionError("I R3_REDUNDANT must have rank 3 and 5 vertices.")
    if rank_of_codes(common_core) != K_BITS - 1:
        raise AssertionError("I common evaluation core must have rank 3.")
    if not set(common_core).issubset(set(r3_red_support)):
        raise AssertionError("I common core must be supported in R3_REDUNDANT.")
    if not set(common_core).issubset(set(r4_can_support)):
        raise AssertionError("I common core must be supported in R4_CANONICAL.")

    profile_support = {
        "FULL16": full_support,
        "R4_REDUNDANT": r4_red_support,
        "R4_CANONICAL": r4_can_support,
        "R3_REDUNDANT": r3_red_support,
    }
    profile_basis = {
        "FULL16": full_basis,
        "R4_REDUNDANT": full_basis,
        "R4_CANONICAL": full_basis,
        "R3_REDUNDANT": retained_basis,
    }

    profile_supports = {
        profile: {r: tuple(profile_support[profile]) for r in X_REGIONS}
        for profile in I_THEOREM_SAFE_ORDER
    }
    profile_bases = {
        profile: {r: tuple(profile_basis[profile]) for r in X_REGIONS}
        for profile in I_THEOREM_SAFE_ORDER
    }

    # Same total n and equal 25% region shares in every I profile.
    common_n = equal_region_total_cap(
        data,
        train_pool,
        region_labels,
        [profile_supports[p] for p in I_THEOREM_SAFE_ORDER],
        max_train_n,
    )

    scenarios: Dict[str, Scenario] = {}
    for p_ix, profile in enumerate(I_THEOREM_SAFE_ORDER):
        support = tuple(profile_support[profile])
        basis = tuple(profile_basis[profile])
        scenario = build_equal_region_scenario(
            data,
            train_pool,
            population_eval_idx,
            region_labels,
            I_THEOREM_SAFE_SCENARIO_NAMES[profile],
            profile_supports[profile],
            profile_bases[profile],
            common_n,
            seed + 10_000 + 1000 * p_ix,
            (
                "Theorem-facing supported-only ladder: all four X regions are "
                "represented equally and share the same local treatment support; "
                f"profile={profile}. Evaluation is restricted to observed support."
            ),
            "I_THEOREM_SAFE_SUPPORT_LADDER",
        )
        scenario.metadata.update({
            "i_profile": profile,
            "evaluation_mode": "supported_only",
            "evaluation_support_codes": support,
            "evaluation_support_bits": tuple(bit_string(c) for c in support),
            "common_evaluation_codes": common_core,
            "common_evaluation_bits": tuple(bit_string(c) for c in common_core),
            "missing_component_index": int(missing_component),
            "omitted_direction_code": int(omitted_direction),
            "omitted_direction_bits": bit_string(omitted_direction),
            "redundant_vertex_code": int(redundant_vertex),
            "redundant_vertex_bits": bit_string(redundant_vertex),
            "retained_basis_codes": retained_basis,
            "retained_basis_bits": tuple(bit_string(c) for c in retained_basis),
            "support_rank": int(rank_of_codes(support)),
            "support_cardinality": int(len(support)),
        })
        scenarios[scenario.name] = scenario

    return scenarios, {
        "common_train_n": int(common_n),
        "profile_supports": profile_supports,
        "profile_bases": profile_bases,
        "common_core": common_core,
        "common_core_bits": tuple(bit_string(c) for c in common_core),
        "missing_component_index": int(missing_component),
        "omitted_direction_code": int(omitted_direction),
        "omitted_direction_bits": bit_string(omitted_direction),
        "redundant_vertex_code": int(redundant_vertex),
        "redundant_vertex_bits": bit_string(redundant_vertex),
        "retained_basis": retained_basis,
        "retained_basis_bits": tuple(bit_string(c) for c in retained_basis),
    }


def build_all_regional_scenarios(
    data: DGPData,
    train_pool: np.ndarray,
    population_eval_idx: np.ndarray,
    region_labels: np.ndarray,
    rotation: int,
    seed: int,
    dominant_share: float,
    max_train_n: int | None,
) -> Tuple[Dict[str, Scenario], Dict[str, object]]:
    rank_map = cyclic_rank_map(rotation)

    E_supports, E_bases = supports_for_rank_map_canonical(rank_map)
    F_supports, F_bases = supports_for_rank_map_general(
        data, train_pool, region_labels, rank_map, seed + 100, redundant=False
    )
    G_supports, G_bases = supports_for_rank_map_general(
        data, train_pool, region_labels, rank_map, seed + 200, redundant=True
    )

    # E and F are directly comparable: matched support cardinality. Use same n.
    n_E = regional_family_cap(
        data, train_pool, region_labels, E_supports, E_bases, rank_map, dominant_share, max_train_n
    )
    n_F = regional_family_cap(
        data, train_pool, region_labels, F_supports, F_bases, rank_map, dominant_share, max_train_n
    )
    n_EF = min(n_E, n_F)
    n_G = regional_family_cap(
        data, train_pool, region_labels, G_supports, G_bases, rank_map, dominant_share, max_train_n
    )

    E = build_regional_family(
        data, train_pool, population_eval_idx, region_labels, rank_map,
        E_supports, E_bases, "E_CANONICAL", "canonical local support",
        n_EF, dominant_share, seed + 10000,
    )
    F = build_regional_family(
        data, train_pool, population_eval_idx, region_labels, rank_map,
        F_supports, F_bases, "F_GENERAL_BASIS", "arbitrary basis support with matched cardinality",
        n_EF, dominant_share, seed + 20000,
    )
    G = build_regional_family(
        data, train_pool, population_eval_idx, region_labels, rank_map,
        G_supports, G_bases, "G_GENERAL_REDUNDANT", "arbitrary rank-preserving support with redundant vertices",
        n_G, dominant_share, seed + 30000,
    )

    return {**E, **F, **G}, {
        "rank_map": rank_map,
        "E_supports": E_supports,
        "F_supports": F_supports,
        "G_supports": G_supports,
        "E_bases": E_bases,
        "F_bases": F_bases,
        "G_bases": G_bases,
        "common_train_n_EF": n_EF,
        "train_n_G": n_G,
        "rotation": rotation,
    }


def build_selected_family_scenarios(
    data: DGPData,
    train_pool: np.ndarray,
    population_eval_idx: np.ndarray,
    region_labels: np.ndarray,
    rotation: int,
    seed: int,
    dominant_share: float,
    max_train_n: int | None,
    selected_families: Sequence[str],
    cycle_index: int,
) -> Tuple[Dict[str, Scenario], Dict[str, object]]:
    """Build only the requested comparison closure, not the entire suite.

    Besides reducing preparation time, this prevents a feasibility problem in
    an unrelated family from aborting a partial run.  The returned metadata is
    keyed by the public CLI family names (plus ``regional`` for E/F/G).
    """
    selected = set(map(str, selected_families))
    scenarios: Dict[str, Scenario] = {}
    metadata: Dict[str, object] = {}

    # BASE belongs to every partial comparison closure and is also the anchor
    # source from which all K experiments remove rows.
    full2k: Scenario | None = None
    if "BASE" in selected or selected.intersection(K_SUBFAMILIES):
        full2k = build_full_2k_given_x_baseline(
            data,
            train_pool,
            population_eval_idx,
            region_labels,
            seed + 45_000,
            max_train_n,
        )
        if "BASE" in selected:
            scenarios[FULL2K_BASELINE_NAME] = full2k

    if "ABC" in selected:
        abc, abc_meta = build_abc_scenarios(
            data,
            train_pool,
            population_eval_idx,
            region_labels,
            seed + 50_000,
            max_train_n,
        )
        scenarios.update(abc)
        metadata["ABC"] = abc_meta

    if "D" in selected:
        dsc, d_meta = build_d_scenarios(
            data,
            train_pool,
            population_eval_idx,
            region_labels,
            seed + 60_000,
            max_train_n,
        )
        scenarios.update(dsc)
        metadata["D"] = d_meta

    if selected.intersection(("E", "F", "G")):
        regional, regional_meta = build_all_regional_scenarios(
            data,
            train_pool,
            population_eval_idx,
            region_labels,
            rotation,
            seed + 70_000,
            dominant_share,
            max_train_n,
        )
        scenarios.update(
            {
                name: scenario
                for name, scenario in regional.items()
                if scenario_family_selector(scenario) in selected
            }
        )
        metadata["regional"] = regional_meta

    # H is built before J because the latter reuses two H scenarios as exact
    # endpoints.  L requests H as its external support-ladder comparator.
    support_ladder: Dict[str, Scenario] = {}
    support_ladder_meta: Mapping[str, object] | None = None
    if "H" in selected or "J" in selected:
        support_ladder, support_ladder_meta = build_local_support_ladder_scenarios(
            data,
            train_pool,
            population_eval_idx,
            region_labels,
            rotation,
            seed + 80_000,
            dominant_share,
            max_train_n,
            cycle_index=cycle_index,
        )
        if "H" in selected:
            scenarios.update(support_ladder)
        metadata["H"] = support_ladder_meta

    if "J" in selected:
        if support_ladder_meta is None:
            raise AssertionError("J dependency closure did not include H.")
        j_smoothing, j_meta = build_rank4_support_smoothing_scenarios(
            data,
            train_pool,
            population_eval_idx,
            region_labels,
            seed + 80_000,
            support_ladder,
            support_ladder_meta,
        )
        scenarios.update(j_smoothing)
        metadata["J"] = j_meta

    if "L" in selected:
        l_norm, l_meta = build_l_rank4_norm_richness_scenarios(
            data,
            train_pool,
            population_eval_idx,
            region_labels,
            rotation,
            seed + 75_000,
            dominant_share,
            max_train_n,
        )
        scenarios.update(l_norm)
        metadata["L"] = l_meta

    if "I" in selected:
        theorem_safe, i_meta = build_theorem_safe_support_ladder_scenarios(
            data,
            train_pool,
            population_eval_idx,
            region_labels,
            seed + 90_000,
            max_train_n,
            cycle_index=cycle_index,
        )
        scenarios.update(theorem_safe)
        metadata["I"] = i_meta

    if selected.intersection(K_SUBFAMILIES):
        if full2k is None:
            raise AssertionError("K dependency closure did not construct BASE anchors.")

        if "K_NEIGHBOR" in selected:
            neighbor, neighbor_meta = build_neighbor_support_extrapolation_scenarios(
                data, full2k, region_labels, seed + 100_000
            )
            scenarios.update(neighbor)
            metadata["K_NEIGHBOR"] = neighbor_meta

        if "K_MOD5" in selected:
            k_mod5, k_mod5_meta = build_k_unsupported_mod5_scenarios(
                data, full2k, region_labels, seed + 110_000
            )
            scenarios.update(k_mod5)
            metadata["K_MOD5"] = k_mod5_meta

        k_random: Dict[str, Scenario] = {}
        k_random_meta: Mapping[str, object] | None = None
        if selected.intersection(("K_RANDOM", "K_RANDOM3", "K_EXTREME")):
            k_random, k_random_meta = build_k_random_mod8_scenarios(
                data, full2k, region_labels, seed + 120_000
            )
            if "K_RANDOM" in selected:
                scenarios.update(k_random)
            metadata["K_RANDOM"] = k_random_meta

        if "K_RANDOM3" in selected:
            if k_random_meta is None:
                raise AssertionError("K_RANDOM3 dependency closure omitted K_RANDOM.")
            k_random3, k_random3_meta = build_k_random3_mod8_scenario(
                data, k_random, k_random_meta, seed + 120_000
            )
            scenarios.update(k_random3)
            metadata["K_RANDOM3"] = k_random3_meta

        if "K_EXTREME" in selected:
            if k_random_meta is None:
                raise AssertionError("K_EXTREME dependency closure omitted K_RANDOM.")
            k_extreme, k_extreme_meta = build_k_extreme_scenarios(
                data, k_random, k_random_meta, seed + 130_000
            )
            scenarios.update(k_extreme)
            metadata["K_EXTREME"] = k_extreme_meta

    if not scenarios:
        raise RuntimeError(
            f"Family selection {tuple(selected_families)} produced no scenarios."
        )
    return scenarios, metadata


# =====================================================================
# Support diagnostics
# =====================================================================

def scenario_support_diagnostics_rows(
    data: DGPData,
    scenario: Scenario,
    region_labels: np.ndarray,
) -> List[Dict[str, object]]:
    rows = []

    source_all = scenario.training_source_idx()
    t_all = scenario.training_t(data)

    for region in X_REGIONS:
        m_region = region_labels[source_all] == region
        source_r = source_all[m_region]
        t_r = t_all[m_region]

        obs = tuple(sorted(np.unique(t_r).astype(int).tolist())) if len(t_r) else tuple()
        basis = tuple(map(int, scenario.local_bases.get(region, tuple())))
        intended = tuple(map(int, scenario.local_supports.get(region, tuple())))
        geom = geometry_diagnostics(obs)

        counts = np.bincount(t_r, minlength=N_REGIMES) if len(t_r) else np.zeros(N_REGIMES, dtype=int)
        baseline_share = float(counts[0] / len(t_r)) if len(t_r) else 0.0
        basis_shares = [float(counts[t] / len(t_r)) for t in basis] if len(t_r) else []
        epsilon_u_hat = min(basis_shares) if basis_shares else 0.0
        all16 = int(len(obs) == N_REGIMES)
        supported_shares = [float(counts[t] / len(t_r)) for t in obs] if len(t_r) else []
        min_supported_share = min(supported_shares) if supported_shares else 0.0

        # Exact per-x empirical support diagnostic. In ordinary observational
        # scenarios each X appears once, so this is one. In the fully-crossed
        # reference each X appears under all 16 actions.
        if len(source_r):
            unique_x, inverse = np.unique(source_r, return_inverse=True)
            n_t_per_x = []
            for j in range(len(unique_x)):
                n_t_per_x.append(len(np.unique(t_r[inverse == j])))
            min_t_per_x = int(min(n_t_per_x))
            max_t_per_x = int(max(n_t_per_x))
            all_2k_per_x = int(all(v == N_REGIMES for v in n_t_per_x))
        else:
            unique_x = np.array([], dtype=int)
            min_t_per_x = 0
            max_t_per_x = 0
            all_2k_per_x = 0

        rows.append({
            "seed": data.seed,
            "scenario": scenario.name,
            "family": scenario.family,
            "x_region": region,
            "n_train_region_rows": int(len(t_r)),
            "n_unique_x_region": int(len(unique_x)),
            "min_treatments_per_x": min_t_per_x,
            "max_treatments_per_x": max_t_per_x,
            "all_2k_per_x": all_2k_per_x,
            "intended_support": ",".join(map(str, intended)),
            "intended_support_bits": " | ".join(bit_strings(intended)),
            "basis": ",".join(map(str, basis)),
            "basis_bits": " | ".join(bit_strings(basis)),
            "observed_support": ",".join(map(str, obs)),
            "observed_support_bits": " | ".join(bit_strings(obs)),
            "baseline_share_hat": baseline_share,
            "epsilon_U_hat": epsilon_u_hat,
            "min_supported_vertex_share_hat": min_supported_share,
            "all_16_vertices_present": all16,
            **geom,
        })
    return rows

# =====================================================================
# Model interface
# =====================================================================

class PotentialOutcomeEstimator:
    def fit(self, X: np.ndarray, A_bits: np.ndarray, y: np.ndarray) -> "PotentialOutcomeEstimator":
        raise NotImplementedError

    def predict_potential(self, X: np.ndarray, regimes_bits: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def treatment_codes_from_bits(bits: np.ndarray) -> np.ndarray:
    """Convert little-endian treatment bits to categorical regime indices."""
    bits = np.asarray(bits)
    if bits.ndim != 2 or bits.shape[1] != K_BITS:
        raise ValueError(
            f"Treatment matrix must have shape (n, {K_BITS}); received {bits.shape}."
        )
    if not np.all(np.isfinite(bits)):
        raise ValueError("Treatment matrix contains NaN/Inf.")
    rounded = np.rint(bits).astype(np.int64)
    if not np.allclose(bits, rounded) or not np.all(np.isin(rounded, (0, 1))):
        raise ValueError("Treatment columns must be binary (0/1).")
    powers = 2 ** np.arange(K_BITS, dtype=np.int64)
    return (rounded @ powers).astype(np.int64)


class ModelsPyPotentialOutcomeAdapter(PotentialOutcomeEstimator):
    """Bridge this benchmark's bit interface to the sibling models.py."""

    def __init__(self, model: Any, name: str):
        self.model = model
        self.name = name

    def fit(
        self,
        X: np.ndarray,
        A_bits: np.ndarray,
        y: np.ndarray,
    ) -> "ModelsPyPotentialOutcomeAdapter":
        # All four public wrappers accept categorical indices.  Using one
        # representation for every model avoids name-based special cases.
        treatment_codes = treatment_codes_from_bits(A_bits)
        self.model.fit(X, treatment_codes, y)
        return self

    def predict_potential(self, X: np.ndarray, regimes_bits: np.ndarray) -> np.ndarray:
        requested_codes = treatment_codes_from_bits(regimes_bits)
        predictions = np.asarray(
            self.model.predict_potential_outcomes(X), dtype=float
        )
        expected_shape = (len(X), N_REGIMES)
        if predictions.shape != expected_shape:
            raise ValueError(
                f"{self.name} returned shape {predictions.shape}; expected {expected_shape}."
            )
        if not np.all(np.isfinite(predictions)):
            raise ValueError(f"{self.name} returned predictions with NaN/Inf.")
        # This DGP's target is a Bernoulli conditional mean.  The previous
        # in-file estimators also clipped to the probability range; keeping the
        # clip matters particularly for HydraNet's unconstrained MSE heads.
        return np.clip(predictions[:, requested_codes], 0.0, 1.0)


def build_model_factories(
    config: Mapping[str, Any],
    selected_models: Sequence[str],
    device: str,
) -> Dict[str, Callable[[int], PotentialOutcomeEstimator]]:
    """Create fresh models.py estimators for each model/scenario fit."""
    requested = set(selected_models)
    unknown = requested.difference(AVAILABLE_MODELS)
    if unknown:
        raise ValueError(f"Unknown model(s): {sorted(unknown)}")

    rf_n_jobs = int(config.get("rf_n_jobs", RF_N_JOBS_DEFAULT))
    if rf_n_jobs == 0 or rf_n_jobs < -1:
        raise ValueError("rf_n_jobs must be -1 or a positive integer.")

    factories: Dict[str, Callable[[int], PotentialOutcomeEstimator]] = {}

    if "S-Learner_RF" in requested:
        def make_s_learner(seed: int) -> PotentialOutcomeEstimator:
            rf = make_random_forest(
                random_state=seed,
                n_estimators=int(config.get("rf_trees", 300)),
                min_samples_leaf=int(config.get("rf_min_samples_leaf", 20)),
                n_jobs=rf_n_jobs,
            )
            model = SLearnerMulti(base_model=rf, K=N_REGIMES, name="S-Learner_RF")
            return ModelsPyPotentialOutcomeAdapter(model, "S-Learner_RF")

        factories["S-Learner_RF"] = make_s_learner

    if "T-Learner_RF" in requested:
        def make_t_learner(seed: int) -> PotentialOutcomeEstimator:
            rf = make_random_forest(
                random_state=seed,
                n_estimators=int(config.get("rf_trees", 300)),
                min_samples_leaf=int(config.get("rf_min_samples_leaf", 20)),
                n_jobs=rf_n_jobs,
            )
            model = TLearnerMulti(base_model=rf, K=N_REGIMES, name="T-Learner_RF")
            return ModelsPyPotentialOutcomeAdapter(model, "T-Learner_RF")

        factories["T-Learner_RF"] = make_t_learner

    if "HydraNet" in requested:
        def make_hydranet(seed: int) -> PotentialOutcomeEstimator:
            hydra_config = HydraNetConfig(
                hidden=int(config.get("hydra_hidden", 128)),
                depth=int(config.get("hydra_depth", 2)),
                dropout=float(config.get("hydra_dropout", 0.0)),
                epochs=int(config.get("hydra_epochs", 200)),
                batch_size=int(config.get("hydra_batch_size", 256)),
                lr=float(config.get("hydra_lr", 1e-3)),
                weight_decay=float(config.get("hydra_weight_decay", 1e-4)),
                alpha_propensity=float(config.get("hydra_alpha_propensity", 1.0)),
                verbose=bool(config.get("verbose_hydra", False)),
                device=device,
                random_state=seed,
            )
            model = HydraNetCausalModel(
                K=N_REGIMES, config=hydra_config, name="HydraNet"
            )
            return ModelsPyPotentialOutcomeAdapter(model, "HydraNet")

        factories["HydraNet"] = make_hydranet

    if "MEDUSA" in requested:
        def make_medusa(seed: int) -> PotentialOutcomeEstimator:
            medusa_config = MedusaConfig(
                latent_dim=int(config.get("medusa_latent_dim", 12)),
                treatment_bits=K_BITS,
                epochs=int(config.get("medusa_epochs", 100)),
                batch_size=int(config.get("medusa_batch_size", 256)),
                lr=float(config.get("medusa_lr", 8e-4)),
                weight_decay=float(config.get("medusa_weight_decay", 2e-4)),
                alpha_propensity=float(config.get("medusa_alpha_propensity", 0.3)),
                beta_anchor=float(config.get("medusa_beta_anchor", 0.05)),
                gradient_clip=float(config.get("medusa_gradient_clip", 5.0)),
                verbose=bool(config.get("verbose_medusa", False)),
                device=device,
                random_state=seed,
            )
            model = MedusaCausalModel(
                K=N_REGIMES, config=medusa_config, name="MEDUSA"
            )
            return ModelsPyPotentialOutcomeAdapter(model, "MEDUSA")

        factories["MEDUSA"] = make_medusa

    return {name: factories[name] for name in AVAILABLE_MODELS if name in factories}


# =====================================================================
# Metrics aligned to mu(x,t)
# =====================================================================

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def cate_metrics(mu_true: np.ndarray, mu_pred: np.ndarray) -> Dict[str, float]:
    true_tau = mu_true[:, 1:] - mu_true[:, [0]]
    pred_tau = mu_pred[:, 1:] - mu_pred[:, [0]]
    err = pred_tau - true_tau
    return {
        "cate_rmse_origin": float(np.sqrt(np.mean(err**2))),
        "cate_mae_origin": float(np.mean(np.abs(err))),
        "cate_bias_origin": float(np.mean(err)),
    }


def population_ate_metrics(mu_true: np.ndarray, mu_pred: np.ndarray) -> Dict[str, float]:
    true_tau = np.mean(mu_true[:, 1:] - mu_true[:, [0]], axis=0)
    pred_tau = np.mean(mu_pred[:, 1:] - mu_pred[:, [0]], axis=0)
    err = pred_tau - true_tau
    return {
        "mean_ate_abs_error": float(np.mean(np.abs(err))),
        "mean_ate_rmse": float(np.sqrt(np.mean(err**2))),
        "mean_ate_bias": float(np.mean(err)),
    }


def policy_metrics(mu_true: np.ndarray, mu_pred: np.ndarray) -> Dict[str, float]:
    """Policy quality plus explicit treatment-recovery metrics.

    ``policy_accuracy`` and ``treatment_recovery_accuracy`` are numerically
    identical: both require exact recovery of the optimal treatment regime.
    The duplicate name is intentional.  ``policy_accuracy`` is retained for
    backward compatibility, while ``treatment_recovery_*`` makes the clinical
    interpretation explicit: did the model identify the actual intervention?

    Hamming metrics exploit the structured binary treatment vector.  They
    distinguish, for example, predicting 1000 instead of 1001 (one component
    wrong) from predicting 0110 instead of 1001 (four components wrong).
    """
    true_opt = np.argmax(mu_true, axis=1)
    pred_opt = np.argmax(mu_pred, axis=1)
    rows = np.arange(len(mu_true))
    opt_value = np.max(mu_true, axis=1)
    chosen_value = mu_true[rows, pred_opt]
    regret = opt_value - chosen_value
    wrong = pred_opt != true_opt

    true_bits = REGIME_BITS[true_opt].astype(int)
    pred_bits = REGIME_BITS[pred_opt].astype(int)
    bit_mismatch = true_bits != pred_bits
    hamming_distance_i = np.sum(bit_mismatch, axis=1)
    hamming_loss_i = np.mean(bit_mismatch, axis=1)
    exact_recovery = ~wrong

    return {
        "policy_accuracy": float(np.mean(exact_recovery)),
        "policy_error_rate": float(np.mean(wrong)),
        "policy_regret": float(np.mean(regret)),
        "policy_regret_when_wrong": float(np.mean(regret[wrong])) if np.any(wrong) else 0.0,
        "policy_value": float(np.mean(chosen_value)),
        "optimal_policy_value": float(np.mean(opt_value)),
        # Treatment recovery: exact intervention identification.
        "treatment_recovery_accuracy": float(np.mean(exact_recovery)),
        "treatment_recovery_error_rate": float(np.mean(~exact_recovery)),
        # Structured recovery: how many treatment components were wrong?
        "treatment_recovery_hamming_distance": float(np.mean(hamming_distance_i)),
        "treatment_recovery_hamming_loss": float(np.mean(hamming_loss_i)),
        "treatment_recovery_bit_accuracy": float(1.0 - np.mean(hamming_loss_i)),
    }



def subset_cate_metrics(
    mu_true: np.ndarray,
    mu_pred: np.ndarray,
    treatment_codes: Sequence[int],
) -> Dict[str, float]:
    """CATE metrics restricted to an explicitly supported treatment set."""
    codes = tuple(map(int, treatment_codes))
    if ORIGIN not in codes:
        raise ValueError("Supported-only CATE evaluation requires baseline 0000.")
    b = codes.index(ORIGIN)
    nonbase = [j for j, c in enumerate(codes) if c != ORIGIN]
    if not nonbase:
        return {
            "cate_rmse_origin": np.nan,
            "cate_mae_origin": np.nan,
            "cate_bias_origin": np.nan,
        }
    true_tau = mu_true[:, nonbase] - mu_true[:, [b]]
    pred_tau = mu_pred[:, nonbase] - mu_pred[:, [b]]
    err = pred_tau - true_tau
    return {
        "cate_rmse_origin": float(np.sqrt(np.mean(err**2))),
        "cate_mae_origin": float(np.mean(np.abs(err))),
        "cate_bias_origin": float(np.mean(err)),
    }


def subset_ate_metrics(
    mu_true: np.ndarray,
    mu_pred: np.ndarray,
    treatment_codes: Sequence[int],
) -> Dict[str, float]:
    """ATE metrics restricted to supported actions relative to baseline."""
    codes = tuple(map(int, treatment_codes))
    if ORIGIN not in codes:
        raise ValueError("Supported-only ATE evaluation requires baseline 0000.")
    b = codes.index(ORIGIN)
    nonbase = [j for j, c in enumerate(codes) if c != ORIGIN]
    if not nonbase:
        return {
            "mean_ate_abs_error": np.nan,
            "mean_ate_rmse": np.nan,
            "mean_ate_bias": np.nan,
        }
    true_tau = np.mean(mu_true[:, nonbase] - mu_true[:, [b]], axis=0)
    pred_tau = np.mean(mu_pred[:, nonbase] - mu_pred[:, [b]], axis=0)
    err = pred_tau - true_tau
    return {
        "mean_ate_abs_error": float(np.mean(np.abs(err))),
        "mean_ate_rmse": float(np.sqrt(np.mean(err**2))),
        "mean_ate_bias": float(np.mean(err)),
    }


def subset_policy_metrics(
    mu_true: np.ndarray,
    mu_pred: np.ndarray,
    treatment_codes: Sequence[int],
) -> Dict[str, float]:
    """Policy/recovery metrics where the action set is exactly treatment_codes."""
    codes = np.asarray(tuple(map(int, treatment_codes)), dtype=int)
    if mu_true.shape[1] != len(codes) or mu_pred.shape[1] != len(codes):
        raise ValueError("Subset policy arrays do not match treatment_codes.")
    true_pos = np.argmax(mu_true, axis=1)
    pred_pos = np.argmax(mu_pred, axis=1)
    rows = np.arange(len(mu_true))
    opt_value = mu_true[rows, true_pos]
    chosen_value = mu_true[rows, pred_pos]
    regret = opt_value - chosen_value
    true_codes = codes[true_pos]
    pred_codes = codes[pred_pos]
    exact = true_codes == pred_codes
    wrong = ~exact
    true_bits = REGIME_BITS[true_codes].astype(int)
    pred_bits = REGIME_BITS[pred_codes].astype(int)
    mismatch = true_bits != pred_bits
    hamming = np.sum(mismatch, axis=1)
    hamming_loss = np.mean(mismatch, axis=1)
    return {
        "policy_accuracy": float(np.mean(exact)),
        "policy_error_rate": float(np.mean(wrong)),
        "policy_regret": float(np.mean(regret)),
        "policy_regret_when_wrong": float(np.mean(regret[wrong])) if np.any(wrong) else 0.0,
        "policy_value": float(np.mean(chosen_value)),
        "optimal_policy_value": float(np.mean(opt_value)),
        "treatment_recovery_accuracy": float(np.mean(exact)),
        "treatment_recovery_error_rate": float(np.mean(wrong)),
        "treatment_recovery_hamming_distance": float(np.mean(hamming)),
        "treatment_recovery_hamming_loss": float(np.mean(hamming_loss)),
        "treatment_recovery_bit_accuracy": float(1.0 - np.mean(hamming_loss)),
    }


def subset_metric_bundle(
    mu_true: np.ndarray,
    mu_pred: np.ndarray,
    treatment_codes: Sequence[int],
) -> Dict[str, float]:
    """All theorem-safe metrics on an explicitly supported action matrix."""
    out = {"rmse_mu": rmse(mu_true, mu_pred)}
    out.update(subset_cate_metrics(mu_true, mu_pred, treatment_codes))
    out.update(subset_ate_metrics(mu_true, mu_pred, treatment_codes))
    out.update(subset_policy_metrics(mu_true, mu_pred, treatment_codes))
    return out



def mod8_specialized_metric_dict(
    data: DGPData,
    group_idx: Mapping[int, np.ndarray],
    pred_by_group: Mapping[int, np.ndarray],
    support_codes: Mapping[int, Sequence[int]],
    prefix: str,
) -> Dict[str, object]:
    """Specialized all-16/supported/unsupported metrics for a modulo-8 design.

    This helper is shared by K-RANDOM4 and K-RANDOM3 reference/mixed rows.  The
    evaluation X partition is fixed; ``support_codes`` only defines which actions
    count as locally observed versus removed for the support-specific diagnostics.
    """
    all_codes = tuple(range(N_REGIMES))
    ordered_mods = tuple(range(8))
    if set(group_idx) != set(ordered_mods) or set(support_codes) != set(ordered_mods):
        raise RuntimeError(f"{prefix}: expected exactly modulo classes 0..7.")

    true_all = np.concatenate([data.mu_conditional[np.asarray(group_idx[m], dtype=int)] for m in ordered_mods], axis=0)
    pred_all = np.concatenate([np.asarray(pred_by_group[m], dtype=float) for m in ordered_mods], axis=0)
    overall_all = subset_metric_bundle(true_all, pred_all, all_codes)
    out: Dict[str, object] = {
        **{f"{prefix}_overall_all16_{k}": v for k, v in overall_all.items()}
    }

    unsupported_mu_true: List[np.ndarray] = []
    unsupported_mu_pred: List[np.ndarray] = []
    unsupported_tau_err: List[np.ndarray] = []
    unsupported_ate_abs: List[float] = []

    for mod in ordered_mods:
        idx = np.asarray(group_idx[mod], dtype=int)
        pred = np.asarray(pred_by_group[mod], dtype=float)
        true = data.mu_conditional[idx]
        support = tuple(sorted(map(int, support_codes[mod])))
        if ORIGIN not in support:
            raise RuntimeError(f"{prefix}: origin missing from support for mod {mod}.")

        all_bundle = subset_metric_bundle(true, pred, all_codes)
        out.update({f"{prefix}_m{mod}_all16_{k}": v for k, v in all_bundle.items()})

        sup_arr = np.asarray(support, dtype=int)
        sup_bundle = subset_metric_bundle(true[:, sup_arr], pred[:, sup_arr], support)
        out.update({f"{prefix}_m{mod}_supported_{k}": v for k, v in sup_bundle.items()})

        unsupported = tuple(c for c in all_codes if c not in set(support))
        if unsupported:
            uarr = np.asarray(unsupported, dtype=int)
            u_true = true[:, uarr]
            u_pred = pred[:, uarr]
            unsupported_mu_true.append(u_true.reshape(-1))
            unsupported_mu_pred.append(u_pred.reshape(-1))

            true_tau_u = u_true - true[:, [ORIGIN]]
            pred_tau_u = u_pred - pred[:, [ORIGIN]]
            err_tau_u = pred_tau_u - true_tau_u
            unsupported_tau_err.append(err_tau_u.reshape(-1))
            true_ate_u = np.mean(true_tau_u, axis=0)
            pred_ate_u = np.mean(pred_tau_u, axis=0)
            ate_abs = float(np.mean(np.abs(pred_ate_u - true_ate_u)))
            unsupported_ate_abs.append(ate_abs)

            out[f"{prefix}_m{mod}_unsupported_rmse_mu"] = rmse(u_true, u_pred)
            out[f"{prefix}_m{mod}_unsupported_cate_rmse_origin"] = float(
                np.sqrt(np.mean(err_tau_u**2))
            )
            out[f"{prefix}_m{mod}_unsupported_mean_ate_abs_error"] = ate_abs
        else:
            out[f"{prefix}_m{mod}_unsupported_rmse_mu"] = np.nan
            out[f"{prefix}_m{mod}_unsupported_cate_rmse_origin"] = np.nan
            out[f"{prefix}_m{mod}_unsupported_mean_ate_abs_error"] = np.nan

        out[f"{prefix}_m{mod}_n_x"] = int(len(idx))
        out[f"{prefix}_m{mod}_support_rank"] = int(rank_of_codes(support))
        out[f"{prefix}_m{mod}_support_cardinality"] = int(len(support))
        out[f"{prefix}_m{mod}_support_bits"] = ",".join(bit_string(c) for c in support)

    if unsupported_mu_true:
        u_true_flat = np.concatenate(unsupported_mu_true)
        u_pred_flat = np.concatenate(unsupported_mu_pred)
        u_tau_err_flat = np.concatenate(unsupported_tau_err)
        out[f"{prefix}_overall_unsupported_rmse_mu"] = rmse(u_true_flat, u_pred_flat)
        out[f"{prefix}_overall_unsupported_cate_rmse_origin"] = float(
            np.sqrt(np.mean(u_tau_err_flat**2))
        )
        out[f"{prefix}_overall_unsupported_mean_ate_abs_error"] = float(
            np.mean(unsupported_ate_abs)
        )
    else:
        out[f"{prefix}_overall_unsupported_rmse_mu"] = np.nan
        out[f"{prefix}_overall_unsupported_cate_rmse_origin"] = np.nan
        out[f"{prefix}_overall_unsupported_mean_ate_abs_error"] = np.nan

    return out



def variable_anchor_support_metric_dict(
    data: DGPData,
    idx: np.ndarray,
    pred_all16: np.ndarray,
    support_by_anchor: Mapping[int, Sequence[int]],
    prefix: str,
) -> Dict[str, object]:
    """Metrics for a design whose local support can vary at every X anchor.

    Unlike MOD8 diagnostics, this deliberately avoids one output column per
    support matrix.  K-EXTREME may use hundreds of distinct supports, so the
    stable summary is overall plus rank-specific.  This keeps seed_metrics and
    summary.csv compact and avoids pandas fragmentation.
    """
    idx = np.asarray(idx, dtype=int)
    pred = np.asarray(pred_all16, dtype=float)
    if pred.shape != (len(idx), N_REGIMES):
        raise ValueError(f"{prefix}: expected predictions {(len(idx), N_REGIMES)}, got {pred.shape}.")
    true = data.mu_conditional[idx]
    all_codes = tuple(range(N_REGIMES))
    out: Dict[str, object] = {
        **{f"{prefix}_overall_all16_{k}": v for k, v in subset_metric_bundle(true, pred, all_codes).items()}
    }

    support_mask = np.zeros((len(idx), N_REGIMES), dtype=bool)
    ranks = np.zeros(len(idx), dtype=int)
    cards = np.zeros(len(idx), dtype=int)
    for i, x in enumerate(idx):
        support = tuple(sorted(map(int, support_by_anchor[int(x)])))
        if ORIGIN not in support:
            raise RuntimeError(f"{prefix}: origin missing at x={int(x)}.")
        support_mask[i, np.asarray(support, dtype=int)] = True
        ranks[i] = rank_of_codes(support)
        cards[i] = len(support)

    out[f"{prefix}_overall_supported_rmse_mu"] = rmse(true[support_mask], pred[support_mask])
    unsupported_mask = ~support_mask
    out[f"{prefix}_overall_unsupported_rmse_mu"] = (
        rmse(true[unsupported_mask], pred[unsupported_mask]) if np.any(unsupported_mask) else np.nan
    )

    tau_true = true - true[:, [ORIGIN]]
    tau_pred = pred - pred[:, [ORIGIN]]
    tau_err = tau_pred - tau_true
    unsupported_nonorigin = unsupported_mask.copy()
    unsupported_nonorigin[:, ORIGIN] = False
    out[f"{prefix}_overall_unsupported_cate_rmse_origin"] = (
        float(np.sqrt(np.mean(tau_err[unsupported_nonorigin] ** 2)))
        if np.any(unsupported_nonorigin) else np.nan
    )

    ate_abs: List[float] = []
    for t in range(1, N_REGIMES):
        rows = unsupported_nonorigin[:, t]
        if not np.any(rows):
            continue
        ate_abs.append(abs(float(np.mean(tau_pred[rows, t]) - np.mean(tau_true[rows, t]))))
    out[f"{prefix}_overall_unsupported_mean_ate_abs_error"] = (
        float(np.mean(ate_abs)) if ate_abs else np.nan
    )
    out[f"{prefix}_mean_local_rank"] = float(np.mean(ranks))
    out[f"{prefix}_mean_local_cardinality"] = float(np.mean(cards))

    for r in sorted(set(map(int, ranks))):
        rows = ranks == r
        bundle = subset_metric_bundle(true[rows], pred[rows], all_codes)
        out.update({f"{prefix}_rank{r}_all16_{k}": v for k, v in bundle.items()})
        rank_support_mask = support_mask[rows]
        rank_true = true[rows]
        rank_pred = pred[rows]
        rank_tau_err = tau_err[rows]
        out[f"{prefix}_rank{r}_n_x"] = int(np.sum(rows))
        out[f"{prefix}_rank{r}_supported_rmse_mu"] = rmse(
            rank_true[rank_support_mask], rank_pred[rank_support_mask]
        )
        rank_unsup = ~rank_support_mask
        out[f"{prefix}_rank{r}_unsupported_rmse_mu"] = (
            rmse(rank_true[rank_unsup], rank_pred[rank_unsup]) if np.any(rank_unsup) else np.nan
        )
        rank_unsup[:, ORIGIN] = False
        out[f"{prefix}_rank{r}_unsupported_cate_rmse_origin"] = (
            float(np.sqrt(np.mean(rank_tau_err[rank_unsup] ** 2))) if np.any(rank_unsup) else np.nan
        )

    return out


def local_support_entry_mask(
    idx: np.ndarray,
    region_labels: np.ndarray,
    local_supports: Mapping[str, Sequence[int]],
) -> np.ndarray:
    """[n_eval,16] mask: whether treatment t is locally supported for that X stratum."""
    mask = np.zeros((len(idx), N_REGIMES), dtype=bool)
    for region in X_REGIONS:
        rows = np.where(region_labels[idx] == region)[0]
        if not len(rows):
            continue
        support = tuple(map(int, local_supports.get(region, tuple())))
        if support:
            mask[np.ix_(rows, np.asarray(support, dtype=int))] = True
    return mask


def target_surface_metrics(
    mu_true: np.ndarray,
    mu_pred: np.ndarray,
    support_mask: np.ndarray,
    realized_true: np.ndarray | None = None,
) -> Dict[str, float]:
    out = {
        "rmse_mu_all": rmse(mu_true, mu_pred),
    }
    if np.any(support_mask):
        out["rmse_mu_supported_local"] = rmse(mu_true[support_mask], mu_pred[support_mask])
    else:
        out["rmse_mu_supported_local"] = np.nan
    if np.any(~support_mask):
        out["rmse_mu_unsupported_local"] = rmse(mu_true[~support_mask], mu_pred[~support_mask])
    else:
        out["rmse_mu_unsupported_local"] = np.nan

    out.update(cate_metrics(mu_true, mu_pred))
    out.update(policy_metrics(mu_true, mu_pred))

    if realized_true is not None:
        out["stress_rmse_realized_potential"] = rmse(realized_true, mu_pred)
    return out


def detailed_rows(
    mu_true: np.ndarray,
    mu_pred: np.ndarray,
    seed: int,
    model: str,
    scenario: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    ate_rows, cate_rows, policy_rows = [], [], []
    true_tau = mu_true[:, 1:] - mu_true[:, [0]]
    pred_tau = mu_pred[:, 1:] - mu_pred[:, [0]]

    for t in range(1, N_REGIMES):
        et = pred_tau[:, t-1] - true_tau[:, t-1]
        ate_true = float(np.mean(true_tau[:, t-1]))
        ate_pred = float(np.mean(pred_tau[:, t-1]))
        common = {
            "seed": seed, "model": model, "scenario": scenario,
            "reference_t": 0, "t": t, "t_bits": bit_string(t),
        }
        ate_rows.append({
            **common,
            "ate_true": ate_true,
            "ate_pred": ate_pred,
            "ate_error": ate_pred - ate_true,
            "ate_abs_error": abs(ate_pred - ate_true),
        })
        cate_rows.append({
            **common,
            "cate_rmse": float(np.sqrt(np.mean(et**2))),
            "cate_mae": float(np.mean(np.abs(et))),
            "cate_bias": float(np.mean(et)),
        })

    true_opt = np.argmax(mu_true, axis=1)
    pred_opt = np.argmax(mu_pred, axis=1)
    rows = np.arange(len(mu_true))
    regret_i = np.max(mu_true, axis=1) - mu_true[rows, pred_opt]
    for t in range(N_REGIMES):
        m = true_opt == t
        if not np.any(m):
            continue
        true_bits_t = REGIME_BITS[t].astype(int)
        pred_bits_m = REGIME_BITS[pred_opt[m]].astype(int)
        class_hamming = np.sum(pred_bits_m != true_bits_t[None, :], axis=1)
        policy_rows.append({
            "seed": seed, "model": model, "scenario": scenario,
            "true_optimal_t": t,
            "true_optimal_t_bits": bit_string(t),
            "n_population_eval_optimal_t": int(np.sum(m)),
            "policy_class_accuracy": float(np.mean(pred_opt[m] == t)),
            "policy_class_mean_regret": float(np.mean(regret_i[m])),
            # Explicit treatment-recovery diagnostics for this optimal class.
            "treatment_recovery_class_accuracy": float(np.mean(pred_opt[m] == t)),
            "treatment_recovery_class_mean_hamming_distance": float(np.mean(class_hamming)),
            "treatment_recovery_class_bit_accuracy": float(1.0 - np.mean(class_hamming) / K_BITS),
        })
    return ate_rows, cate_rows, policy_rows


def evaluate_one_scenario(
    data: DGPData,
    scenario: Scenario,
    population_eval_idx: np.ndarray,
    region_labels: np.ndarray,
    model_name: str,
    model_factory: Callable[[int], PotentialOutcomeEstimator],
    model_seed: int,
    training_outcome: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    tr = scenario.train_idx
    te = scenario.test_idx

    if scenario.has_training_override():
        src = scenario.training_source_idx()
        t_train = scenario.training_t(data)
        X_train = data.X[src]
        A_train = REGIME_BITS[t_train]
        if training_outcome == "contaminated":
            y_train = np.asarray(scenario.train_y_contaminated_override, dtype=float)
        else:
            y_train = np.asarray(scenario.train_y_clean_override, dtype=float)
    else:
        src = np.asarray(tr, dtype=int)
        X_train = data.X[tr]
        A_train = data.A[tr]
        y_train = data.Yobs[tr] if training_outcome == "contaminated" else data.Y_factual_clean[tr]

    total_started = time.perf_counter()
    model = model_factory(model_seed)
    fit_started = time.perf_counter()
    model.fit(X_train, A_train, y_train)
    fit_seconds = time.perf_counter() - fit_started

    # K family: evaluate the same weak/even X anchors in both matched scenarios.
    # The reference observes all 16 actions at those X values; the mixed design
    # observes only the canonical 5 there and must borrow noncanonical treatment
    # information from adjacent full-support X anchors.
    if scenario.family == "K_NEIGHBOR_SUPPORT_EXTRAPOLATION":
        weak_idx = np.asarray(scenario.metadata["k_weak_anchor_idx"], dtype=int)
        full_idx = np.asarray(scenario.metadata["k_full_anchor_idx"], dtype=int)
        canonical_codes = tuple(map(int, scenario.metadata["k_weak_support_codes"]))
        noncanonical_codes = tuple(map(int, scenario.metadata["k_noncanonical_codes"]))
        all_codes = tuple(range(N_REGIMES))

        target_predict_started = time.perf_counter()
        pred_weak = model.predict_potential(data.X[weak_idx], REGIME_BITS)
        pred_full = model.predict_potential(data.X[full_idx], REGIME_BITS)
        target_predict_seconds = time.perf_counter() - target_predict_started
        population_predict_seconds = 0.0

        true_weak = data.mu_conditional[weak_idx]
        true_full = data.mu_conditional[full_idx]
        weak_all = subset_metric_bundle(true_weak, pred_weak, all_codes)
        full_all = subset_metric_bundle(true_full, pred_full, all_codes)

        can_arr = np.asarray(canonical_codes, dtype=int)
        weak_can = subset_metric_bundle(
            true_weak[:, can_arr], pred_weak[:, can_arr], canonical_codes
        )

        joint_arr = np.asarray(noncanonical_codes, dtype=int)
        if len(joint_arr) != N_REGIMES - len(canonical_codes):
            raise RuntimeError("K noncanonical action count mismatch.")
        weak_unsupported_rmse = rmse(true_weak[:, joint_arr], pred_weak[:, joint_arr])
        true_tau_u = true_weak[:, joint_arr] - true_weak[:, [ORIGIN]]
        pred_tau_u = pred_weak[:, joint_arr] - pred_weak[:, [ORIGIN]]
        err_tau_u = pred_tau_u - true_tau_u
        weak_unsupported_cate = float(np.sqrt(np.mean(err_tau_u**2)))
        true_ate_u = np.mean(true_tau_u, axis=0)
        pred_ate_u = np.mean(pred_tau_u, axis=0)
        weak_unsupported_ate = float(np.mean(np.abs(pred_ate_u - true_ate_u)))

        # Do not mix K's anchor-target metrics with the ordinary population metrics.
        # K has its own explicit k_* columns and paired contrasts.
        nan_standard = {
            "target_rmse_mu_all": np.nan,
            "target_rmse_mu_supported_local": np.nan,
            "target_rmse_mu_unsupported_local": np.nan,
            "target_cate_rmse_origin": np.nan,
            "target_cate_mae_origin": np.nan,
            "target_cate_bias_origin": np.nan,
            "target_policy_accuracy": np.nan,
            "target_policy_error_rate": np.nan,
            "target_policy_regret": np.nan,
            "target_policy_regret_when_wrong": np.nan,
            "target_policy_value": np.nan,
            "target_optimal_policy_value": np.nan,
            "target_treatment_recovery_accuracy": np.nan,
            "target_treatment_recovery_error_rate": np.nan,
            "target_treatment_recovery_hamming_distance": np.nan,
            "target_treatment_recovery_hamming_loss": np.nan,
            "target_treatment_recovery_bit_accuracy": np.nan,
            "population_mean_ate_abs_error": np.nan,
            "population_mean_ate_rmse": np.nan,
            "population_mean_ate_bias": np.nan,
            "population_cate_rmse_origin": np.nan,
            "population_cate_mae_origin": np.nan,
            "population_cate_bias_origin": np.nan,
            "population_policy_accuracy": np.nan,
            "population_policy_error_rate": np.nan,
            "population_policy_regret": np.nan,
            "population_policy_regret_when_wrong": np.nan,
            "population_policy_value": np.nan,
            "population_optimal_policy_value": np.nan,
            "population_treatment_recovery_accuracy": np.nan,
            "population_treatment_recovery_error_rate": np.nan,
            "population_treatment_recovery_hamming_distance": np.nan,
            "population_treatment_recovery_hamming_loss": np.nan,
            "population_treatment_recovery_bit_accuracy": np.nan,
        }

        row = {
            "seed": data.seed,
            "model": model_name,
            "scenario": scenario.name,
            "family": scenario.family,
            "n_train": scenario.n_training_rows(),
            "n_unique_train_x": scenario.n_unique_training_x(),
            "n_target_eval": len(weak_idx),
            "n_population_eval": len(population_eval_idx),
            "fit_seconds": fit_seconds,
            "target_predict_seconds": target_predict_seconds,
            "population_predict_seconds": population_predict_seconds,
            "total_model_seconds": time.perf_counter() - total_started,
            "global_train_support": ",".join(map(str, scenario.global_observed_support(data))),
            "global_rank_U": rank_of_codes(scenario.global_observed_support(data)),
            "literal_all_2k_per_x": int(scenario.metadata.get("literal_all_2k_per_x", 0)),
            "k_profile": str(scenario.metadata["k_profile"]),
            "k_evaluation_mode": "same_weak_x_all16",
            "k_n_weak_x": int(len(weak_idx)),
            "k_n_full_x": int(len(full_idx)),
            "k_neighbor_s_extra_mean": float(scenario.metadata["k_neighbor_s_extra_mean"]),
            "k_neighbor_s_extra_median": float(scenario.metadata["k_neighbor_s_extra_median"]),
            "k_neighbor_s_extra_p95": float(scenario.metadata["k_neighbor_s_extra_p95"]),
            "k_neighbor_xstar_l2_mean": float(scenario.metadata["k_neighbor_xstar_l2_mean"]),
            "k_neighbor_xstar_l2_median": float(scenario.metadata["k_neighbor_xstar_l2_median"]),
            "k_neighbor_xstar_l2_p95": float(scenario.metadata["k_neighbor_xstar_l2_p95"]),
            **nan_standard,
            **{f"k_weak_all16_{k}": v for k, v in weak_all.items()},
            **{f"k_full_all16_{k}": v for k, v in full_all.items()},
            **{f"k_weak_canonical_{k}": v for k, v in weak_can.items()},
            "k_weak_unsupported_rmse_mu": weak_unsupported_rmse,
            "k_weak_unsupported_cate_rmse_origin": weak_unsupported_cate,
            "k_weak_unsupported_mean_ate_abs_error": weak_unsupported_ate,
        }
        return row, [], [], []

    # K-UNSUPPORTED family: evaluate all 16 treatments separately on the five
    # interleaved modulo-5 X populations.  The mixed scenario progressively loses
    # treatment directions, while the matched reference observes all 16 actions at
    # exactly the same X anchors.
    if scenario.family == "K_UNSUPPORTED_MOD5_EXTRAPOLATION":
        all_codes = tuple(range(N_REGIMES))
        group_idx = {
            str(k): np.asarray(v, dtype=int)
            for k, v in scenario.metadata["ku_group_anchor_idx"].items()
        }
        support_codes = {
            str(k): tuple(map(int, v))
            for k, v in scenario.metadata["ku_support_codes"].items()
        }

        target_predict_started = time.perf_counter()
        pred_by_group = {
            label: model.predict_potential(data.X[idx], REGIME_BITS)
            for label, idx in group_idx.items()
        }
        target_predict_seconds = time.perf_counter() - target_predict_started
        population_predict_seconds = 0.0

        nan_standard = {
            "target_rmse_mu_all": np.nan,
            "target_rmse_mu_supported_local": np.nan,
            "target_rmse_mu_unsupported_local": np.nan,
            "target_cate_rmse_origin": np.nan,
            "target_cate_mae_origin": np.nan,
            "target_cate_bias_origin": np.nan,
            "target_policy_accuracy": np.nan,
            "target_policy_error_rate": np.nan,
            "target_policy_regret": np.nan,
            "target_policy_regret_when_wrong": np.nan,
            "target_policy_value": np.nan,
            "target_optimal_policy_value": np.nan,
            "target_treatment_recovery_accuracy": np.nan,
            "target_treatment_recovery_error_rate": np.nan,
            "target_treatment_recovery_hamming_distance": np.nan,
            "target_treatment_recovery_hamming_loss": np.nan,
            "target_treatment_recovery_bit_accuracy": np.nan,
            "population_mean_ate_abs_error": np.nan,
            "population_mean_ate_rmse": np.nan,
            "population_mean_ate_bias": np.nan,
            "population_cate_rmse_origin": np.nan,
            "population_cate_mae_origin": np.nan,
            "population_cate_bias_origin": np.nan,
            "population_policy_accuracy": np.nan,
            "population_policy_error_rate": np.nan,
            "population_policy_regret": np.nan,
            "population_policy_regret_when_wrong": np.nan,
            "population_policy_value": np.nan,
            "population_optimal_policy_value": np.nan,
            "population_treatment_recovery_accuracy": np.nan,
            "population_treatment_recovery_error_rate": np.nan,
            "population_treatment_recovery_hamming_distance": np.nan,
            "population_treatment_recovery_hamming_loss": np.nan,
            "population_treatment_recovery_bit_accuracy": np.nan,
        }

        row: Dict[str, object] = {
            "seed": data.seed,
            "model": model_name,
            "scenario": scenario.name,
            "family": scenario.family,
            "n_train": scenario.n_training_rows(),
            "n_unique_train_x": scenario.n_unique_training_x(),
            "n_target_eval": int(sum(len(v) for v in group_idx.values())),
            "n_population_eval": len(population_eval_idx),
            "fit_seconds": fit_seconds,
            "target_predict_seconds": target_predict_seconds,
            "population_predict_seconds": population_predict_seconds,
            "total_model_seconds": time.perf_counter() - total_started,
            "global_train_support": ",".join(map(str, scenario.global_observed_support(data))),
            "global_rank_U": rank_of_codes(scenario.global_observed_support(data)),
            "literal_all_2k_per_x": int(scenario.metadata.get("literal_all_2k_per_x", 0)),
            "ku_profile": str(scenario.metadata["ku_profile"]),
            "ku_evaluation_mode": "five_interleaved_x_groups_all16",
            "ku_neighbor_s_extra_mean": float(scenario.metadata["ku_neighbor_s_extra_mean"]),
            "ku_neighbor_s_extra_median": float(scenario.metadata["ku_neighbor_s_extra_median"]),
            "ku_neighbor_s_extra_p95": float(scenario.metadata["ku_neighbor_s_extra_p95"]),
            "ku_neighbor_xstar_l2_mean": float(scenario.metadata["ku_neighbor_xstar_l2_mean"]),
            "ku_neighbor_xstar_l2_median": float(scenario.metadata["ku_neighbor_xstar_l2_median"]),
            "ku_neighbor_xstar_l2_p95": float(scenario.metadata["ku_neighbor_xstar_l2_p95"]),
            **nan_standard,
        }

        for label in ("full", "r4", "r3", "r2", "r1"):
            idx = group_idx[label]
            pred = pred_by_group[label]
            true = data.mu_conditional[idx]
            all_bundle = subset_metric_bundle(true, pred, all_codes)
            row.update({f"ku_{label}_all16_{k}": v for k, v in all_bundle.items()})

            supported = tuple(support_codes[label])
            sup_arr = np.asarray(supported, dtype=int)
            sup_bundle = subset_metric_bundle(true[:, sup_arr], pred[:, sup_arr], supported)
            row.update({f"ku_{label}_supported_{k}": v for k, v in sup_bundle.items()})

            unsupported = tuple(c for c in all_codes if c not in set(supported))
            if unsupported:
                uarr = np.asarray(unsupported, dtype=int)
                row[f"ku_{label}_unsupported_rmse_mu"] = rmse(true[:, uarr], pred[:, uarr])
                true_tau_u = true[:, uarr] - true[:, [ORIGIN]]
                pred_tau_u = pred[:, uarr] - pred[:, [ORIGIN]]
                err_tau_u = pred_tau_u - true_tau_u
                row[f"ku_{label}_unsupported_cate_rmse_origin"] = float(
                    np.sqrt(np.mean(err_tau_u**2))
                )
                true_ate_u = np.mean(true_tau_u, axis=0)
                pred_ate_u = np.mean(pred_tau_u, axis=0)
                row[f"ku_{label}_unsupported_mean_ate_abs_error"] = float(
                    np.mean(np.abs(pred_ate_u - true_ate_u))
                )
            else:
                row[f"ku_{label}_unsupported_rmse_mu"] = np.nan
                row[f"ku_{label}_unsupported_cate_rmse_origin"] = np.nan
                row[f"ku_{label}_unsupported_mean_ate_abs_error"] = np.nan

            row[f"ku_{label}_n_x"] = int(len(idx))
            row[f"ku_{label}_support_rank"] = int(rank_of_codes(supported))
            row[f"ku_{label}_support_cardinality"] = int(len(supported))

        return row, [], [], []

    # K-RANDOM family: eight interleaved minimal rank-4 supports with maximally
    # diverse treatment bases.  Both matched scenarios are evaluated on all 16
    # potential outcomes at exactly the same X anchors.
    if scenario.family == "K_RANDOM_MOD8_RANK4_EXTRAPOLATION":
        all_codes = tuple(range(N_REGIMES))
        group_idx = {
            int(k): np.asarray(v, dtype=int)
            for k, v in scenario.metadata["kr_group_anchor_idx"].items()
        }
        support_codes = {
            int(k): tuple(map(int, v))
            for k, v in scenario.metadata["kr_support_codes"].items()
        }

        target_predict_started = time.perf_counter()
        pred_by_group = {
            mod: model.predict_potential(data.X[idx], REGIME_BITS)
            for mod, idx in group_idx.items()
        }
        target_predict_seconds = time.perf_counter() - target_predict_started
        population_predict_seconds = 0.0

        nan_standard = {
            "target_rmse_mu_all": np.nan,
            "target_rmse_mu_supported_local": np.nan,
            "target_rmse_mu_unsupported_local": np.nan,
            "target_cate_rmse_origin": np.nan,
            "target_cate_mae_origin": np.nan,
            "target_cate_bias_origin": np.nan,
            "target_policy_accuracy": np.nan,
            "target_policy_error_rate": np.nan,
            "target_policy_regret": np.nan,
            "target_policy_regret_when_wrong": np.nan,
            "target_policy_value": np.nan,
            "target_optimal_policy_value": np.nan,
            "target_treatment_recovery_accuracy": np.nan,
            "target_treatment_recovery_error_rate": np.nan,
            "target_treatment_recovery_hamming_distance": np.nan,
            "target_treatment_recovery_hamming_loss": np.nan,
            "target_treatment_recovery_bit_accuracy": np.nan,
            "population_mean_ate_abs_error": np.nan,
            "population_mean_ate_rmse": np.nan,
            "population_mean_ate_bias": np.nan,
            "population_cate_rmse_origin": np.nan,
            "population_cate_mae_origin": np.nan,
            "population_cate_bias_origin": np.nan,
            "population_policy_accuracy": np.nan,
            "population_policy_error_rate": np.nan,
            "population_policy_regret": np.nan,
            "population_policy_regret_when_wrong": np.nan,
            "population_policy_value": np.nan,
            "population_optimal_policy_value": np.nan,
            "population_treatment_recovery_accuracy": np.nan,
            "population_treatment_recovery_error_rate": np.nan,
            "population_treatment_recovery_hamming_distance": np.nan,
            "population_treatment_recovery_hamming_loss": np.nan,
            "population_treatment_recovery_bit_accuracy": np.nan,
        }

        # Overall all-16 metric on the union of the eight modulo populations.
        ordered_mods = tuple(range(8))
        true_all = np.concatenate([data.mu_conditional[group_idx[m]] for m in ordered_mods], axis=0)
        pred_all = np.concatenate([pred_by_group[m] for m in ordered_mods], axis=0)
        overall_all = subset_metric_bundle(true_all, pred_all, all_codes)

        unsupported_mu_true: List[np.ndarray] = []
        unsupported_mu_pred: List[np.ndarray] = []
        unsupported_tau_err: List[np.ndarray] = []
        unsupported_ate_abs: List[float] = []

        row: Dict[str, object] = {
            "seed": data.seed,
            "model": model_name,
            "scenario": scenario.name,
            "family": scenario.family,
            "n_train": scenario.n_training_rows(),
            "n_unique_train_x": scenario.n_unique_training_x(),
            "n_target_eval": int(sum(len(v) for v in group_idx.values())),
            "n_population_eval": len(population_eval_idx),
            "fit_seconds": fit_seconds,
            "target_predict_seconds": target_predict_seconds,
            "population_predict_seconds": population_predict_seconds,
            "total_model_seconds": time.perf_counter() - total_started,
            "global_train_support": ",".join(map(str, scenario.global_observed_support(data))),
            "global_rank_U": rank_of_codes(scenario.global_observed_support(data)),
            "literal_all_2k_per_x": int(scenario.metadata.get("literal_all_2k_per_x", 0)),
            "kr_profile": str(scenario.metadata["kr_profile"]),
            "kr_evaluation_mode": "eight_interleaved_rank4_supports_all16",
            "kr_max_pairwise_nonzero_overlap": int(scenario.metadata["kr_max_pairwise_nonzero_overlap"]),
            "kr_mean_pairwise_nonzero_overlap": float(scenario.metadata["kr_mean_pairwise_nonzero_overlap"]),
            "kr_internal_hamming_diversity_min": int(scenario.metadata["kr_internal_hamming_diversity_min"]),
            "kr_neighbor_s_extra_mean": float(scenario.metadata["kr_neighbor_s_extra_mean"]),
            "kr_neighbor_s_extra_median": float(scenario.metadata["kr_neighbor_s_extra_median"]),
            "kr_neighbor_s_extra_p95": float(scenario.metadata["kr_neighbor_s_extra_p95"]),
            "kr_neighbor_xstar_l2_mean": float(scenario.metadata["kr_neighbor_xstar_l2_mean"]),
            "kr_neighbor_xstar_l2_median": float(scenario.metadata["kr_neighbor_xstar_l2_median"]),
            "kr_neighbor_xstar_l2_p95": float(scenario.metadata["kr_neighbor_xstar_l2_p95"]),
            **nan_standard,
            **{f"kr_overall_all16_{k}": v for k, v in overall_all.items()},
            # Neutral MOD8 aliases allow direct K-RANDOM3 vs K-RANDOM4 contrasts.
            **{f"kmod8_overall_all16_{k}": v for k, v in overall_all.items()},
            # The exact same ordered-X anchors are reused by K-EXTREME.
            **{f"kext_overall_all16_{k}": v for k, v in overall_all.items()},
        }

        for mod in ordered_mods:
            idx = group_idx[mod]
            pred = pred_by_group[mod]
            true = data.mu_conditional[idx]
            support = tuple(support_codes[mod])
            if len(support) != 5 or rank_of_codes(support) != K_BITS:
                raise RuntimeError(f"K-RANDOM mod {mod} evaluation support is not rank4/size5.")

            all_bundle = subset_metric_bundle(true, pred, all_codes)
            row.update({f"kr_m{mod}_all16_{k}": v for k, v in all_bundle.items()})

            sup_arr = np.asarray(support, dtype=int)
            sup_bundle = subset_metric_bundle(true[:, sup_arr], pred[:, sup_arr], support)
            row.update({f"kr_m{mod}_supported_{k}": v for k, v in sup_bundle.items()})

            unsupported = tuple(c for c in all_codes if c not in set(support))
            uarr = np.asarray(unsupported, dtype=int)
            u_true = true[:, uarr]
            u_pred = pred[:, uarr]
            unsupported_mu_true.append(u_true.reshape(-1))
            unsupported_mu_pred.append(u_pred.reshape(-1))

            true_tau_u = u_true - true[:, [ORIGIN]]
            pred_tau_u = u_pred - pred[:, [ORIGIN]]
            err_tau_u = pred_tau_u - true_tau_u
            unsupported_tau_err.append(err_tau_u.reshape(-1))
            true_ate_u = np.mean(true_tau_u, axis=0)
            pred_ate_u = np.mean(pred_tau_u, axis=0)
            ate_abs = float(np.mean(np.abs(pred_ate_u - true_ate_u)))
            unsupported_ate_abs.append(ate_abs)

            row[f"kr_m{mod}_unsupported_rmse_mu"] = rmse(u_true, u_pred)
            row[f"kr_m{mod}_unsupported_cate_rmse_origin"] = float(np.sqrt(np.mean(err_tau_u**2)))
            row[f"kr_m{mod}_unsupported_mean_ate_abs_error"] = ate_abs
            row[f"kr_m{mod}_n_x"] = int(len(idx))
            row[f"kr_m{mod}_support_rank"] = int(rank_of_codes(support))
            row[f"kr_m{mod}_support_cardinality"] = int(len(support))
            row[f"kr_m{mod}_support_bits"] = ",".join(bit_string(c) for c in support)

        u_true_flat = np.concatenate(unsupported_mu_true)
        u_pred_flat = np.concatenate(unsupported_mu_pred)
        u_tau_err_flat = np.concatenate(unsupported_tau_err)
        row["kr_overall_unsupported_rmse_mu"] = rmse(u_true_flat, u_pred_flat)
        row["kr_overall_unsupported_cate_rmse_origin"] = float(np.sqrt(np.mean(u_tau_err_flat**2)))
        row["kr_overall_unsupported_mean_ate_abs_error"] = float(np.mean(unsupported_ate_abs))

        # The shared full16 reference is also the matched reference for
        # K-RANDOM3.  Reuse the already-computed predictions and emit kr3_*
        # metrics under the rank3/rank2 conceptual support partition without
        # fitting the reference a second time.
        if "kr3_support_codes" in scenario.metadata:
            kr3_support_codes = {
                int(k): tuple(map(int, v))
                for k, v in scenario.metadata["kr3_support_codes"].items()
            }
            row.update({
                "kr3_profile": str(scenario.metadata.get("kr3_profile", "FULL16_REFERENCE")),
                "kr3_evaluation_mode": str(scenario.metadata["kr3_evaluation_mode"]),
                "kr3_max_pairwise_nonzero_overlap": int(scenario.metadata["kr3_max_pairwise_nonzero_overlap"]),
                "kr3_mean_pairwise_nonzero_overlap": float(scenario.metadata["kr3_mean_pairwise_nonzero_overlap"]),
                "kr3_total_hamming_diversity": int(scenario.metadata["kr3_total_hamming_diversity"]),
                "kr3_neighbor_s_extra_mean": float(scenario.metadata["kr3_neighbor_s_extra_mean"]),
                "kr3_neighbor_s_extra_median": float(scenario.metadata["kr3_neighbor_s_extra_median"]),
                "kr3_neighbor_s_extra_p95": float(scenario.metadata["kr3_neighbor_s_extra_p95"]),
                "kr3_neighbor_xstar_l2_mean": float(scenario.metadata["kr3_neighbor_xstar_l2_mean"]),
                "kr3_neighbor_xstar_l2_median": float(scenario.metadata["kr3_neighbor_xstar_l2_median"]),
                "kr3_neighbor_xstar_l2_p95": float(scenario.metadata["kr3_neighbor_xstar_l2_p95"]),
            })
            row.update(
                mod8_specialized_metric_dict(
                    data, group_idx, pred_by_group, kr3_support_codes, prefix="kr3"
                )
            )

        return row, [], [], []

    # K-RANDOM3 family: same ordered-X modulo-8 partition as K-RANDOM4, but
    # every local support is rank 3 or rank 2.  The eight supports collectively
    # cover full16, while no individual target X has a rank-4 training matrix.
    if scenario.family == "K_RANDOM3_MOD8_RANK32_EXTRAPOLATION":
        group_idx = {
            int(k): np.asarray(v, dtype=int)
            for k, v in scenario.metadata["kr3_group_anchor_idx"].items()
        }
        support_codes = {
            int(k): tuple(map(int, v))
            for k, v in scenario.metadata["kr3_support_codes"].items()
        }

        target_predict_started = time.perf_counter()
        pred_by_group = {
            mod: model.predict_potential(data.X[idx], REGIME_BITS)
            for mod, idx in group_idx.items()
        }
        target_predict_seconds = time.perf_counter() - target_predict_started
        population_predict_seconds = 0.0

        for mod in range(8):
            rank = rank_of_codes(support_codes[mod])
            expected_rank = int(scenario.metadata["kr3_rank_by_mod"][mod])
            expected_size = int(scenario.metadata["kr3_cardinality_by_mod"][mod])
            if rank != expected_rank or len(support_codes[mod]) != expected_size:
                raise RuntimeError(
                    f"K-RANDOM3 mod {mod} geometry mismatch: rank={rank}, "
                    f"|S|={len(support_codes[mod])}, expected rank={expected_rank}, |S|={expected_size}."
                )
            if rank >= K_BITS:
                raise RuntimeError("K-RANDOM3 must never evaluate a local rank-4 training support.")

        nan_standard = {
            "target_rmse_mu_all": np.nan,
            "target_rmse_mu_supported_local": np.nan,
            "target_rmse_mu_unsupported_local": np.nan,
            "target_cate_rmse_origin": np.nan,
            "target_cate_mae_origin": np.nan,
            "target_cate_bias_origin": np.nan,
            "target_policy_accuracy": np.nan,
            "target_policy_error_rate": np.nan,
            "target_policy_regret": np.nan,
            "target_policy_regret_when_wrong": np.nan,
            "target_policy_value": np.nan,
            "target_optimal_policy_value": np.nan,
            "target_treatment_recovery_accuracy": np.nan,
            "target_treatment_recovery_error_rate": np.nan,
            "target_treatment_recovery_hamming_distance": np.nan,
            "target_treatment_recovery_hamming_loss": np.nan,
            "target_treatment_recovery_bit_accuracy": np.nan,
            "population_mean_ate_abs_error": np.nan,
            "population_mean_ate_rmse": np.nan,
            "population_mean_ate_bias": np.nan,
            "population_cate_rmse_origin": np.nan,
            "population_cate_mae_origin": np.nan,
            "population_cate_bias_origin": np.nan,
            "population_policy_accuracy": np.nan,
            "population_policy_error_rate": np.nan,
            "population_policy_regret": np.nan,
            "population_policy_regret_when_wrong": np.nan,
            "population_policy_value": np.nan,
            "population_optimal_policy_value": np.nan,
            "population_treatment_recovery_accuracy": np.nan,
            "population_treatment_recovery_error_rate": np.nan,
            "population_treatment_recovery_hamming_distance": np.nan,
            "population_treatment_recovery_hamming_loss": np.nan,
            "population_treatment_recovery_bit_accuracy": np.nan,
        }

        row: Dict[str, object] = {
            "seed": data.seed,
            "model": model_name,
            "scenario": scenario.name,
            "family": scenario.family,
            "n_train": scenario.n_training_rows(),
            "n_unique_train_x": scenario.n_unique_training_x(),
            "n_target_eval": int(sum(len(v) for v in group_idx.values())),
            "n_population_eval": len(population_eval_idx),
            "fit_seconds": fit_seconds,
            "target_predict_seconds": target_predict_seconds,
            "population_predict_seconds": population_predict_seconds,
            "total_model_seconds": time.perf_counter() - total_started,
            "global_train_support": ",".join(map(str, scenario.global_observed_support(data))),
            "global_rank_U": rank_of_codes(scenario.global_observed_support(data)),
            "literal_all_2k_per_x": 0,
            "kr3_profile": str(scenario.metadata["kr3_profile"]),
            "kr3_evaluation_mode": str(scenario.metadata["kr3_evaluation_mode"]),
            "kr3_max_pairwise_nonzero_overlap": int(scenario.metadata["kr3_max_pairwise_nonzero_overlap"]),
            "kr3_mean_pairwise_nonzero_overlap": float(scenario.metadata["kr3_mean_pairwise_nonzero_overlap"]),
            "kr3_total_hamming_diversity": int(scenario.metadata["kr3_total_hamming_diversity"]),
            "kr3_neighbor_s_extra_mean": float(scenario.metadata["kr3_neighbor_s_extra_mean"]),
            "kr3_neighbor_s_extra_median": float(scenario.metadata["kr3_neighbor_s_extra_median"]),
            "kr3_neighbor_s_extra_p95": float(scenario.metadata["kr3_neighbor_s_extra_p95"]),
            "kr3_neighbor_xstar_l2_mean": float(scenario.metadata["kr3_neighbor_xstar_l2_mean"]),
            "kr3_neighbor_xstar_l2_median": float(scenario.metadata["kr3_neighbor_xstar_l2_median"]),
            "kr3_neighbor_xstar_l2_p95": float(scenario.metadata["kr3_neighbor_xstar_l2_p95"]),
            **nan_standard,
        }
        row.update(
            mod8_specialized_metric_dict(
                data, group_idx, pred_by_group, support_codes, prefix="kr3"
            )
        )
        true_all = np.concatenate([data.mu_conditional[group_idx[m]] for m in range(8)], axis=0)
        pred_all = np.concatenate([pred_by_group[m] for m in range(8)], axis=0)
        neutral_all = subset_metric_bundle(true_all, pred_all, tuple(range(N_REGIMES)))
        row.update({f"kmod8_overall_all16_{k}": v for k, v in neutral_all.items()})
        row.update({f"kext_overall_all16_{k}": v for k, v in neutral_all.items()})
        return row, [], [], []

    # K-EXTREME family: maximum number of distinct local support orientations
    # on the exact same ordered-X anchors as K-RANDOM.  The rank4 variant uses
    # all minimal full-rank bases; the rank<4 variant uses all minimal ranks 1-3.
    if scenario.family == "K_EXTREME_MAXMOD_EXTRAPOLATION":
        selected_idx = np.asarray(scenario.metadata["kext_selected_anchor_idx"], dtype=int)
        support_by_anchor = {
            int(k): tuple(map(int, v))
            for k, v in scenario.metadata["kext_anchor_support_codes"].items()
        }
        if set(map(int, selected_idx)) != set(support_by_anchor):
            raise RuntimeError(f"{scenario.name}: K-EXTREME support map does not match evaluation anchors.")

        target_predict_started = time.perf_counter()
        pred = model.predict_potential(data.X[selected_idx], REGIME_BITS)
        target_predict_seconds = time.perf_counter() - target_predict_started
        population_predict_seconds = 0.0

        variant = str(scenario.metadata["kext_variant"])
        prefix = "kext4" if variant == "RANK4_MAXMOD" else "kextlt4"
        metrics_ext = variable_anchor_support_metric_dict(
            data, selected_idx, pred, support_by_anchor, prefix=prefix
        )
        all_bundle = subset_metric_bundle(
            data.mu_conditional[selected_idx], pred, tuple(range(N_REGIMES))
        )

        nan_standard = {
            "target_rmse_mu_all": np.nan,
            "target_rmse_mu_supported_local": np.nan,
            "target_rmse_mu_unsupported_local": np.nan,
            "target_cate_rmse_origin": np.nan,
            "target_cate_mae_origin": np.nan,
            "target_cate_bias_origin": np.nan,
            "target_policy_accuracy": np.nan,
            "target_policy_error_rate": np.nan,
            "target_policy_regret": np.nan,
            "target_policy_regret_when_wrong": np.nan,
            "target_policy_value": np.nan,
            "target_optimal_policy_value": np.nan,
            "target_treatment_recovery_accuracy": np.nan,
            "target_treatment_recovery_error_rate": np.nan,
            "target_treatment_recovery_hamming_distance": np.nan,
            "target_treatment_recovery_hamming_loss": np.nan,
            "target_treatment_recovery_bit_accuracy": np.nan,
            "population_mean_ate_abs_error": np.nan,
            "population_mean_ate_rmse": np.nan,
            "population_mean_ate_bias": np.nan,
            "population_cate_rmse_origin": np.nan,
            "population_cate_mae_origin": np.nan,
            "population_cate_bias_origin": np.nan,
            "population_policy_accuracy": np.nan,
            "population_policy_error_rate": np.nan,
            "population_policy_regret": np.nan,
            "population_policy_regret_when_wrong": np.nan,
            "population_policy_value": np.nan,
            "population_optimal_policy_value": np.nan,
            "population_treatment_recovery_accuracy": np.nan,
            "population_treatment_recovery_error_rate": np.nan,
            "population_treatment_recovery_hamming_distance": np.nan,
            "population_treatment_recovery_hamming_loss": np.nan,
            "population_treatment_recovery_bit_accuracy": np.nan,
        }

        row: Dict[str, object] = {
            "seed": data.seed,
            "model": model_name,
            "scenario": scenario.name,
            "family": scenario.family,
            "n_train": scenario.n_training_rows(),
            "n_unique_train_x": scenario.n_unique_training_x(),
            "n_target_eval": int(len(selected_idx)),
            "n_population_eval": len(population_eval_idx),
            "fit_seconds": fit_seconds,
            "target_predict_seconds": target_predict_seconds,
            "population_predict_seconds": population_predict_seconds,
            "total_model_seconds": time.perf_counter() - total_started,
            "global_train_support": ",".join(map(str, scenario.global_observed_support(data))),
            "global_rank_U": rank_of_codes(scenario.global_observed_support(data)),
            "literal_all_2k_per_x": 0,
            "kext_variant": variant,
            "kext_theoretical_module": int(scenario.metadata["kext_theoretical_module"]),
            "kext_effective_module": int(scenario.metadata["kext_effective_module"]),
            "kext_n_unique_supports_used": int(scenario.metadata["kext_n_unique_supports_used"]),
            "kext_mean_local_rank": float(np.average(
                list(map(int, scenario.metadata["kext_rank_counts"].keys())),
                weights=list(map(int, scenario.metadata["kext_rank_counts"].values())),
            )),
            **nan_standard,
            **metrics_ext,
            **{f"kext_overall_all16_{k}": v for k, v in all_bundle.items()},
        }
        return row, [], [], []

    # I family: theorem-facing supported-only evaluation.  No metric below asks
    # for an unsupported treatment action.  All X strata are represented in
    # training; the treatment matrix itself is reduced before metric evaluation.
    if scenario.family == "I_THEOREM_SAFE_SUPPORT_LADDER":
        native_codes = tuple(map(int, scenario.metadata["evaluation_support_codes"]))
        common_codes = tuple(map(int, scenario.metadata["common_evaluation_codes"]))
        if not set(common_codes).issubset(set(native_codes)):
            raise RuntimeError("I common evaluation core is not contained in native support.")
        for region in X_REGIONS:
            local = set(map(int, scenario.local_supports.get(region, tuple())))
            if not set(native_codes).issubset(local):
                raise RuntimeError(
                    f"{scenario.name}: attempted I evaluation outside support in {region}."
                )

        native_arr = np.asarray(native_codes, dtype=int)
        common_pos = np.asarray([native_codes.index(c) for c in common_codes], dtype=int)

        target_predict_started = time.perf_counter()
        pred_target_native = model.predict_potential(data.X[te], REGIME_BITS[native_arr])
        target_predict_seconds = time.perf_counter() - target_predict_started
        true_target_native = data.mu_conditional[np.ix_(te, native_arr)]
        native_target = subset_metric_bundle(true_target_native, pred_target_native, native_codes)
        common_target = subset_metric_bundle(
            true_target_native[:, common_pos],
            pred_target_native[:, common_pos],
            common_codes,
        )

        population_predict_started = time.perf_counter()
        pred_pop_native = model.predict_potential(
            data.X[population_eval_idx], REGIME_BITS[native_arr]
        )
        population_predict_seconds = time.perf_counter() - population_predict_started
        true_pop_native = data.mu_conditional[np.ix_(population_eval_idx, native_arr)]
        native_pop = subset_metric_bundle(true_pop_native, pred_pop_native, native_codes)
        common_pop = subset_metric_bundle(
            true_pop_native[:, common_pos],
            pred_pop_native[:, common_pos],
            common_codes,
        )

        # Traditional all-16 metrics are deliberately NaN for family I so they
        # cannot be mistaken for theorem-safe quantities.  Use i_common_* for
        # cross-profile comparisons and i_native_* for within-support diagnostics.
        nan_standard = {
            "target_rmse_mu_all": np.nan,
            "target_rmse_mu_supported_local": np.nan,
            "target_rmse_mu_unsupported_local": np.nan,
            "target_cate_rmse_origin": np.nan,
            "target_cate_mae_origin": np.nan,
            "target_cate_bias_origin": np.nan,
            "target_policy_accuracy": np.nan,
            "target_policy_error_rate": np.nan,
            "target_policy_regret": np.nan,
            "target_policy_regret_when_wrong": np.nan,
            "target_policy_value": np.nan,
            "target_optimal_policy_value": np.nan,
            "target_treatment_recovery_accuracy": np.nan,
            "target_treatment_recovery_error_rate": np.nan,
            "target_treatment_recovery_hamming_distance": np.nan,
            "target_treatment_recovery_hamming_loss": np.nan,
            "target_treatment_recovery_bit_accuracy": np.nan,
            "population_mean_ate_abs_error": np.nan,
            "population_mean_ate_rmse": np.nan,
            "population_mean_ate_bias": np.nan,
            "population_cate_rmse_origin": np.nan,
            "population_cate_mae_origin": np.nan,
            "population_cate_bias_origin": np.nan,
            "population_policy_accuracy": np.nan,
            "population_policy_error_rate": np.nan,
            "population_policy_regret": np.nan,
            "population_policy_regret_when_wrong": np.nan,
            "population_policy_value": np.nan,
            "population_optimal_policy_value": np.nan,
            "population_treatment_recovery_accuracy": np.nan,
            "population_treatment_recovery_error_rate": np.nan,
            "population_treatment_recovery_hamming_distance": np.nan,
            "population_treatment_recovery_hamming_loss": np.nan,
            "population_treatment_recovery_bit_accuracy": np.nan,
        }

        row = {
            "seed": data.seed,
            "model": model_name,
            "scenario": scenario.name,
            "family": scenario.family,
            "n_train": scenario.n_training_rows(),
            "n_unique_train_x": scenario.n_unique_training_x(),
            "n_target_eval": len(te),
            "n_population_eval": len(population_eval_idx),
            "fit_seconds": fit_seconds,
            "target_predict_seconds": target_predict_seconds,
            "population_predict_seconds": population_predict_seconds,
            "total_model_seconds": time.perf_counter() - total_started,
            "global_train_support": ",".join(map(str, scenario.global_observed_support(data))),
            "global_rank_U": rank_of_codes(scenario.global_observed_support(data)),
            "literal_all_2k_per_x": int(scenario.metadata.get("literal_all_2k_per_x", 0)),
            "i_evaluation_mode": "supported_only",
            "i_native_support_codes": ",".join(map(str, native_codes)),
            "i_native_support_bits": " | ".join(bit_strings(native_codes)),
            "i_common_support_codes": ",".join(map(str, common_codes)),
            "i_common_support_bits": " | ".join(bit_strings(common_codes)),
            **nan_standard,
            **{f"i_native_{k}": v for k, v in native_pop.items()},
            **{f"i_common_{k}": v for k, v in common_pop.items()},
            **{f"i_target_native_{k}": v for k, v in native_target.items()},
            **{f"i_target_common_{k}": v for k, v in common_target.items()},
        }
        return row, [], [], []

    target_predict_started = time.perf_counter()
    pred_target = model.predict_potential(data.X[te], REGIME_BITS)
    target_predict_seconds = time.perf_counter() - target_predict_started
    support_mask = local_support_entry_mask(te, region_labels, scenario.local_supports)
    target = target_surface_metrics(
        data.mu_conditional[te],
        pred_target,
        support_mask,
        realized_true=data.Ypot_realized[te],
    )

    population_predict_started = time.perf_counter()
    pred_pop = model.predict_potential(data.X[population_eval_idx], REGIME_BITS)
    population_predict_seconds = time.perf_counter() - population_predict_started
    pop_ate = population_ate_metrics(data.mu_conditional[population_eval_idx], pred_pop)
    pop_cate = cate_metrics(data.mu_conditional[population_eval_idx], pred_pop)
    pop_policy = policy_metrics(data.mu_conditional[population_eval_idx], pred_pop)

    row = {
        "seed": data.seed,
        "model": model_name,
        "scenario": scenario.name,
        "family": scenario.family,
        "n_train": scenario.n_training_rows(),
        "n_unique_train_x": scenario.n_unique_training_x(),
        "n_target_eval": len(te),
        "n_population_eval": len(population_eval_idx),
        "fit_seconds": fit_seconds,
        "target_predict_seconds": target_predict_seconds,
        "population_predict_seconds": population_predict_seconds,
        "total_model_seconds": time.perf_counter() - total_started,
        "global_train_support": ",".join(map(str, scenario.global_observed_support(data))),
        "global_rank_U": rank_of_codes(scenario.global_observed_support(data)),
        "literal_all_2k_per_x": int(scenario.metadata.get("literal_all_2k_per_x", 0)),
        **{f"target_{k}": v for k, v in target.items()},
        **{f"population_{k}": v for k, v in pop_ate.items()},
        **{f"population_{k}": v for k, v in pop_cate.items()},
        **{f"population_{k}": v for k, v in pop_policy.items()},
    }

    ate_rows, cate_rows, policy_rows = detailed_rows(
        data.mu_conditional[population_eval_idx],
        pred_pop,
        data.seed,
        model_name,
        scenario.name,
    )
    return row, ate_rows, cate_rows, policy_rows


# =====================================================================
# Statistical summaries
# =====================================================================

def paired_bootstrap_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 9917) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.mean(x)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def one_sided_mean_greater_pvalue(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    sd = np.std(x, ddof=1)
    if sd <= 0:
        return 0.0 if np.mean(x) > 0 else 1.0
    stat = np.mean(x) / (sd / np.sqrt(len(x)))
    return float(student_t.sf(stat, df=len(x)-1))



RELATIVE_METRIC_DIRECTIONS: Dict[str, str] = {
    # lower is better
    "target_rmse_mu_all": "lower",
    "target_cate_rmse_origin": "lower",
    "target_cate_mae_origin": "lower",
    "target_policy_regret": "lower",
    "target_policy_error_rate": "lower",
    "target_treatment_recovery_error_rate": "lower",
    "target_treatment_recovery_hamming_distance": "lower",
    "target_treatment_recovery_hamming_loss": "lower",
    "population_mean_ate_abs_error": "lower",
    "population_mean_ate_rmse": "lower",
    "population_cate_rmse_origin": "lower",
    "population_cate_mae_origin": "lower",
    "population_policy_regret": "lower",
    "population_policy_error_rate": "lower",
    "population_treatment_recovery_error_rate": "lower",
    "population_treatment_recovery_hamming_distance": "lower",
    "population_treatment_recovery_hamming_loss": "lower",
    # higher is better
    "target_policy_accuracy": "higher",
    "target_treatment_recovery_accuracy": "higher",
    "target_treatment_recovery_bit_accuracy": "higher",
    "target_policy_value": "higher",
    "population_policy_accuracy": "higher",
    "population_policy_value": "higher",
    "population_treatment_recovery_accuracy": "higher",
    "population_treatment_recovery_bit_accuracy": "higher",
}


def add_relative_worsening_vs_full2k(metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add seed- and model-matched percentage deterioration relative to the literal
    BASE_FULL_2K_GIVEN_X reference.

    Positive percentage means worse than the fully crossed baseline; negative
    means better. Percentages are left NaN when the matched baseline value is
    numerically zero, while the absolute delta is always retained.
    """
    out = metrics.copy()
    long_rows: List[Dict[str, object]] = []

    base = out[out["scenario"] == FULL2K_BASELINE_NAME].copy()
    if base.empty:
        raise RuntimeError(f"Missing required baseline scenario {FULL2K_BASELINE_NAME}.")

    for metric, direction in RELATIVE_METRIC_DIRECTIONS.items():
        if metric not in out.columns:
            continue

        b = base[["seed", "model", metric]].rename(columns={metric: "base_value"})
        tmp = out[["seed", "model", "scenario", metric]].merge(
            b, on=["seed", "model"], how="left", validate="many_to_one"
        )

        scenario_values = tmp[metric].to_numpy(dtype=float)
        base_values = tmp["base_value"].to_numpy(dtype=float)

        if direction == "lower":
            delta = scenario_values - base_values
        else:
            delta = base_values - scenario_values

        denom = np.abs(base_values)
        pct = np.full(len(tmp), np.nan, dtype=float)
        ok = np.isfinite(denom) & (denom > 1e-12) & np.isfinite(delta)
        pct[ok] = 100.0 * delta[ok] / denom[ok]

        delta_col = f"delta_worse_vs_full2k__{metric}"
        pct_col = f"pct_worse_vs_full2k__{metric}"

        key_delta = {
            (int(seed), str(model), str(scenario)): float(d) if np.isfinite(d) else np.nan
            for seed, model, scenario, d
            in zip(tmp["seed"], tmp["model"], tmp["scenario"], delta)
        }
        key_pct = {
            (int(seed), str(model), str(scenario)): float(v) if np.isfinite(v) else np.nan
            for seed, model, scenario, v
            in zip(tmp["seed"], tmp["model"], tmp["scenario"], pct)
        }

        out[delta_col] = [
            key_delta[(int(seed), str(model), str(scenario))]
            for seed, model, scenario in zip(out["seed"], out["model"], out["scenario"])
        ]
        out[pct_col] = [
            key_pct[(int(seed), str(model), str(scenario))]
            for seed, model, scenario in zip(out["seed"], out["model"], out["scenario"])
        ]

        for i in range(len(tmp)):
            long_rows.append({
                "seed": int(tmp.iloc[i]["seed"]),
                "model": str(tmp.iloc[i]["model"]),
                "scenario": str(tmp.iloc[i]["scenario"]),
                "baseline_scenario": FULL2K_BASELINE_NAME,
                "metric": metric,
                "direction": direction,
                "baseline_value": float(base_values[i]) if np.isfinite(base_values[i]) else np.nan,
                "scenario_value": float(scenario_values[i]) if np.isfinite(scenario_values[i]) else np.nan,
                "absolute_worsening": float(delta[i]) if np.isfinite(delta[i]) else np.nan,
                "pct_worsening_vs_full2k": float(pct[i]) if np.isfinite(pct[i]) else np.nan,
            })

    return out, pd.DataFrame(long_rows)


def build_paired_contrasts(
    metrics: pd.DataFrame,
    n_boot: int = 5000,
) -> pd.DataFrame:
    """Primary directional tests plus descriptive model-based comparisons."""
    error_metrics = [
        "target_rmse_mu_all",
        "target_rmse_mu_unsupported_local",
        "population_mean_ate_abs_error",
        "population_cate_rmse_origin",
        "population_policy_regret",
        "population_policy_error_rate",
        "population_treatment_recovery_error_rate",
        "population_treatment_recovery_hamming_distance",
        # K-RANDOM: eight interleaved, maximally diverse rank-4 size-5 supports.
        "kr_overall_all16_rmse_mu",
        "kr_overall_all16_mean_ate_abs_error",
        "kr_overall_all16_cate_rmse_origin",
        "kr_overall_all16_policy_regret",
        "kr_overall_all16_treatment_recovery_error_rate",
        "kr_overall_all16_treatment_recovery_hamming_distance",
        "kr_overall_unsupported_rmse_mu",
        "kr_overall_unsupported_cate_rmse_origin",
        "kr_overall_unsupported_mean_ate_abs_error",
        # K-RANDOM3: same modulo-8 X partition, but alternating rank-3/rank-2 supports.
        "kr3_overall_all16_rmse_mu",
        "kr3_overall_all16_mean_ate_abs_error",
        "kr3_overall_all16_cate_rmse_origin",
        "kr3_overall_all16_policy_regret",
        "kr3_overall_all16_treatment_recovery_error_rate",
        "kr3_overall_all16_treatment_recovery_hamming_distance",
        "kr3_overall_unsupported_rmse_mu",
        "kr3_overall_unsupported_cate_rmse_origin",
        "kr3_overall_unsupported_mean_ate_abs_error",
        # Neutral all-16 MOD8 metrics shared by K-RANDOM4 and K-RANDOM3.
        "kmod8_overall_all16_rmse_mu",
        "kmod8_overall_all16_mean_ate_abs_error",
        "kmod8_overall_all16_cate_rmse_origin",
        "kmod8_overall_all16_policy_regret",
        "kmod8_overall_all16_treatment_recovery_error_rate",
        "kmod8_overall_all16_treatment_recovery_hamming_distance",
        # K-EXTREME: neutral all-16 metrics on the same ordered-X anchors.
        "kext_overall_all16_rmse_mu",
        "kext_overall_all16_mean_ate_abs_error",
        "kext_overall_all16_cate_rmse_origin",
        "kext_overall_all16_policy_regret",
        "kext_overall_all16_treatment_recovery_error_rate",
        "kext_overall_all16_treatment_recovery_hamming_distance",
        # I family: all cross-profile tests use the fixed, everywhere-supported core.
        "i_common_rmse_mu",
        "i_common_mean_ate_abs_error",
        "i_common_cate_rmse_origin",
        "i_common_policy_regret",
        "i_common_treatment_recovery_error_rate",
        "i_common_treatment_recovery_hamming_distance",
        # K family: same weak/even X target anchors, all-16 and specifically
        # the 11 treatment vertices removed from HALF_CANONICAL.
        "k_weak_all16_rmse_mu",
        "k_weak_all16_mean_ate_abs_error",
        "k_weak_all16_cate_rmse_origin",
        "k_weak_all16_policy_regret",
        "k_weak_all16_treatment_recovery_error_rate",
        "k_weak_all16_treatment_recovery_hamming_distance",
        "k_weak_unsupported_rmse_mu",
        "k_weak_unsupported_cate_rmse_origin",
        "k_weak_unsupported_mean_ate_abs_error",
        # K-UNSUPPORTED: all-16 extrapolation and unsupported-only error for each
        # interleaved rank group.
        "ku_full_all16_rmse_mu",
        "ku_full_all16_cate_rmse_origin",
        "ku_full_all16_policy_regret",
        "ku_full_all16_treatment_recovery_error_rate",
        "ku_r4_all16_rmse_mu",
        "ku_r4_all16_cate_rmse_origin",
        "ku_r4_all16_policy_regret",
        "ku_r4_all16_treatment_recovery_error_rate",
        "ku_r4_unsupported_rmse_mu",
        "ku_r4_unsupported_cate_rmse_origin",
        "ku_r3_all16_rmse_mu",
        "ku_r3_all16_cate_rmse_origin",
        "ku_r3_all16_policy_regret",
        "ku_r3_all16_treatment_recovery_error_rate",
        "ku_r3_unsupported_rmse_mu",
        "ku_r3_unsupported_cate_rmse_origin",
        "ku_r2_all16_rmse_mu",
        "ku_r2_all16_cate_rmse_origin",
        "ku_r2_all16_policy_regret",
        "ku_r2_all16_treatment_recovery_error_rate",
        "ku_r2_unsupported_rmse_mu",
        "ku_r2_unsupported_cate_rmse_origin",
        "ku_r1_all16_rmse_mu",
        "ku_r1_all16_cate_rmse_origin",
        "ku_r1_all16_policy_regret",
        "ku_r1_all16_treatment_recovery_error_rate",
        "ku_r1_unsupported_rmse_mu",
        "ku_r1_unsupported_cate_rmse_origin",
    ]

    directional = [
        ("C_minus_B", "C_RANK3", "B_WEAK"),
        ("Dpoor_minus_Dref", "D_POOR_X", "D_REF_X"),
        (
            "H_R4RED_minus_H_FULL16",
            "H_LOCAL_SUPPORT_LADDER_R4_REDUNDANT_DOM",
            "H_LOCAL_SUPPORT_LADDER_FULL16_DOM",
        ),
        (
            "H_R3RED_minus_H_R4CANONICAL",
            "H_LOCAL_SUPPORT_LADDER_R3_REDUNDANT_DOM",
            "H_LOCAL_SUPPORT_LADDER_R4_CANONICAL_DOM",
        ),
        (
            "H_R3RED_minus_H_R4RED",
            "H_LOCAL_SUPPORT_LADDER_R3_REDUNDANT_DOM",
            "H_LOCAL_SUPPORT_LADDER_R4_REDUNDANT_DOM",
        ),
        (
            "J_SUPPORT12_minus_FULL16",
            "J_RANK4_SUPPORT12_DOM",
            "H_LOCAL_SUPPORT_LADDER_FULL16_DOM",
        ),
        (
            "J_SUPPORT8_minus_SUPPORT12",
            "J_RANK4_SUPPORT8_DOM",
            "J_RANK4_SUPPORT12_DOM",
        ),
        (
            "J_CANONICAL_minus_SUPPORT8",
            "H_LOCAL_SUPPORT_LADDER_R4_CANONICAL_DOM",
            "J_RANK4_SUPPORT8_DOM",
        ),
        (
            "J_CANONICAL_minus_FULL16",
            "H_LOCAL_SUPPORT_LADDER_R4_CANONICAL_DOM",
            "H_LOCAL_SUPPORT_LADDER_FULL16_DOM",
        ),
        (
            "K_HALF_CANONICAL_minus_K_FULL16_REFERENCE",
            "K_NEIGHBOR_HALF_CANONICAL",
            "K_NEIGHBOR_FULL16_REFERENCE",
        ),
        (
            "K_UNSUPPORTED_MOD5_minus_FULL16_REFERENCE",
            "K_UNSUPPORTED_MOD5",
            "K_UNSUPPORTED_FULL16_REFERENCE",
        ),
        (
            "K_RANDOM_MOD8_minus_FULL16_REFERENCE",
            "K_RANDOM_RANK4_MOD8",
            "K_RANDOM_FULL16_REFERENCE",
        ),
        (
            "K_RANDOM3_MOD8_minus_FULL16_REFERENCE",
            "K_RANDOM3_RANK32_MOD8",
            "K_RANDOM_FULL16_REFERENCE",
        ),
        (
            "K_RANDOM3_minus_K_RANDOM4",
            "K_RANDOM3_RANK32_MOD8",
            "K_RANDOM_RANK4_MOD8",
        ),
        (
            "K_EXTREME_R4_minus_FULL16_REFERENCE",
            "K_EXTREME_RANK4_MAXMOD",
            "K_RANDOM_FULL16_REFERENCE",
        ),
        (
            "K_EXTREME_LT4_minus_FULL16_REFERENCE",
            "K_EXTREME_LT4_MAXMOD",
            "K_RANDOM_FULL16_REFERENCE",
        ),
        (
            "K_EXTREME_LT4_minus_K_EXTREME_R4",
            "K_EXTREME_LT4_MAXMOD",
            "K_EXTREME_RANK4_MAXMOD",
        ),
        (
            "K_EXTREME_R4_minus_K_RANDOM4",
            "K_EXTREME_RANK4_MAXMOD",
            "K_RANDOM_RANK4_MOD8",
        ),
        (
            "K_EXTREME_LT4_minus_K_RANDOM3",
            "K_EXTREME_LT4_MAXMOD",
            "K_RANDOM3_RANK32_MOD8",
        ),
        (
            "I_R4RED_minus_I_FULL16",
            "I_THEOREM_SAFE_R4_REDUNDANT",
            "I_THEOREM_SAFE_FULL16",
        ),
        (
            "I_R4CAN_minus_I_R4RED",
            "I_THEOREM_SAFE_R4_CANONICAL",
            "I_THEOREM_SAFE_R4_REDUNDANT",
        ),
        (
            "I_R3RED_minus_I_R4CAN",
            "I_THEOREM_SAFE_R3_REDUNDANT_SUPPORTED_ONLY",
            "I_THEOREM_SAFE_R4_CANONICAL",
        ),
    ]
    descriptive = [
        ("A_minus_FULL2K", "A_STRONG", FULL2K_BASELINE_NAME),
        ("B_minus_FULL2K", "B_WEAK", FULL2K_BASELINE_NAME),
        ("C_minus_FULL2K", "C_RANK3", FULL2K_BASELINE_NAME),
        ("B_minus_A", "B_WEAK", "A_STRONG"),
        (
            "H_R4CANONICAL_minus_H_R4RED",
            "H_LOCAL_SUPPORT_LADDER_R4_CANONICAL_DOM",
            "H_LOCAL_SUPPORT_LADDER_R4_REDUNDANT_DOM",
        ),
        (
            "H_R3RED_minus_H_FULL16",
            "H_LOCAL_SUPPORT_LADDER_R3_REDUNDANT_DOM",
            "H_LOCAL_SUPPORT_LADDER_FULL16_DOM",
        ),
    ]

    for prefix in ("E_CANONICAL", "F_GENERAL_BASIS", "G_GENERAL_REDUNDANT"):
        directional.extend([
            (f"{prefix}_R3dom_minus_R4dom", f"{prefix}_RANK3_DOM", f"{prefix}_RANK4_DOM"),
            (f"{prefix}_R2dom_minus_R3dom", f"{prefix}_RANK2_DOM", f"{prefix}_RANK3_DOM"),
            (f"{prefix}_R1dom_minus_R2dom", f"{prefix}_RANK1_DOM", f"{prefix}_RANK2_DOM"),
            (f"{prefix}_R1dom_minus_R4dom", f"{prefix}_RANK1_DOM", f"{prefix}_RANK4_DOM"),
        ])

    # Matched-cardinality orientation comparison: descriptive, not directional.
    for r in (1, 2, 3, 4):
        descriptive.append((
            f"GENERAL_BASIS_minus_CANONICAL_R{r}dom",
            f"F_GENERAL_BASIS_RANK{r}_DOM",
            f"E_CANONICAL_RANK{r}_DOM",
        ))

    # L keeps rank=4 and |S|=5 everywhere.  These are descriptive contrasts
    # because the geometry theorem does not imply that richer Hamming-weight
    # composition must improve a nonlinear outcome model.
    descriptive.extend([
        (
            "L_NORM12_minus_NORM1",
            L_SCENARIO_NAMES["NORM12"],
            L_SCENARIO_NAMES["NORM1"],
        ),
        (
            "L_NORM123_minus_NORM1",
            L_SCENARIO_NAMES["NORM123"],
            L_SCENARIO_NAMES["NORM1"],
        ),
        (
            "L_NORM1234_minus_NORM1",
            L_SCENARIO_NAMES["NORM1234"],
            L_SCENARIO_NAMES["NORM1"],
        ),
        (
            "L_NORM1234_minus_NORM123",
            L_SCENARIO_NAMES["NORM1234"],
            L_SCENARIO_NAMES["NORM123"],
        ),
    ])

    rows = []
    for model, dfm in metrics.groupby("model"):
        for metric in error_metrics:
            if metric not in dfm.columns:
                continue
            pivot = dfm.pivot_table(index="seed", columns="scenario", values=metric, aggfunc="first")
            for test_type, defs in (("directional", directional), ("descriptive", descriptive)):
                for label, lhs, rhs in defs:
                    # H contains the two endpoints reused by J.  Do not emit a
                    # J-labelled endpoint contrast in an H-only dependency run
                    # unless the two genuinely J-specific middle rungs were fit.
                    if label.startswith("J_") and not {
                        "J_RANK4_SUPPORT12_DOM",
                        "J_RANK4_SUPPORT8_DOM",
                    }.issubset(pivot.columns):
                        continue
                    if lhs not in pivot.columns or rhs not in pivot.columns:
                        continue
                    delta = (pivot[lhs] - pivot[rhs]).dropna().to_numpy()
                    mean, lo, hi = paired_bootstrap_ci(delta, n_boot=n_boot)
                    p = one_sided_mean_greater_pvalue(delta) if test_type == "directional" else np.nan
                    rows.append({
                        "model": model,
                        "contrast": label,
                        "contrast_type": test_type,
                        "metric": metric,
                        "n_pairs": len(delta),
                        "mean_delta": mean,
                        "bootstrap_ci_2.5": lo,
                        "bootstrap_ci_97.5": hi,
                        "one_sided_p_mean_gt_0": p,
                    })
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scenario metrics without propagating DataFrame fragmentation.

    The experiment progressively adds many metric columns (I/K/KU/KR).  Pandas can
    therefore leave ``metrics`` split across a large number of internal blocks.
    Building a compact temporary frame here both removes the PerformanceWarning and
    keeps the summary independent of how the columns were created.
    """
    numeric = [
        c for c in metrics.columns
        if pd.api.types.is_numeric_dtype(metrics[c])
        and (
            c.startswith("target_")
            or c.startswith("population_")
            or c.startswith("i_")
            or c.startswith("k_")
            or c.startswith("ku_")
            or c.startswith("kr_")
            or c.startswith("kr3_")
            or c.startswith("kmod8_")
            or c.startswith("kext_")
            or c.startswith("kext4_")
            or c.startswith("kextlt4_")
            or c.startswith("pct_worse_vs_full2k__")
            or c.startswith("delta_worse_vs_full2k__")
            or c.endswith("_seconds")
            or c in ("n_train", "n_unique_train_x", "global_rank_U")
        )
    ]

    cols = ["model", "scenario", *numeric]
    # Rebuild instead of repeatedly inserting/selecting columns from a fragmented
    # frame.  This is intentionally public-API only (no private pandas internals).
    compact = pd.DataFrame({c: metrics[c].to_numpy(copy=False) for c in cols})
    aggregated = compact.groupby(["model", "scenario"], sort=False)[numeric].agg(
        ["mean", "std", "median"]
    )

    # ``reset_index`` on this very wide MultiIndex frame makes pandas insert
    # the key columns into an already fragmented block manager.  Reconstructing
    # once from arrays is equivalent and stays contiguous.
    result: Dict[str, np.ndarray] = {
        "model": aggregated.index.get_level_values("model").to_numpy(),
        "scenario": aggregated.index.get_level_values("scenario").to_numpy(),
    }
    for metric, statistic in aggregated.columns:
        result[f"{metric}_{statistic}"] = aggregated[(metric, statistic)].to_numpy()
    return pd.DataFrame(result)


# =====================================================================
# Plotting
# =====================================================================

def plot_core(metrics: pd.DataFrame, outpath: Path) -> None:
    order = [FULL2K_BASELINE_NAME, "A_STRONG", "B_WEAK", "C_RANK3"]
    labels = ["Full 2^K|x", "A empirical all-16/stratum", "B weak rank=4", "C rank=3"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, dfm in metrics.groupby("model"):
        y = [dfm.loc[dfm.scenario == s, "target_rmse_mu_all"].mean() for s in order]
        ax.plot(labels, y, marker="o", label=model)
    ax.set_ylabel("RMSE for conditional mean mu(x,t)")
    ax.set_title("Model-based treatment-support validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_regional(metrics: pd.DataFrame, prefix: str, outpath: Path) -> None:
    order = [f"{prefix}_RANK4_DOM", f"{prefix}_RANK3_DOM", f"{prefix}_RANK2_DOM", f"{prefix}_RANK1_DOM"]
    labels = ["rank4 dom", "rank3 dom", "rank2 dom", "rank1 dom"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, dfm in metrics.groupby("model"):
        y = [dfm.loc[dfm.scenario == s, "population_cate_rmse_origin"].mean() for s in order]
        ax.plot(labels, y, marker="o", label=model)
    ax.set_ylabel("Population CATE RMSE vs baseline")
    ax.set_title(prefix.replace("_", " "))
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_l_rank4_norm_richness(metrics: pd.DataFrame, outpath: Path) -> None:
    """Benchmark L: minimal rank-4 supports with increasing treatment-order richness."""
    order = [L_SCENARIO_NAMES[p] for p in L_PROFILE_ORDER]
    labels = ["norm 1", "norm 1-2", "norm 1-3", "norm 1-4"]
    present = set(metrics["scenario"].astype(str))
    if any(s not in present for s in order):
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for model, dfm in metrics.groupby("model"):
        y = np.asarray([
            dfm.loc[dfm.scenario == s, "population_cate_rmse_origin"].mean()
            for s in order
        ], dtype=float)
        if np.all(np.isnan(y)):
            continue
        ax.plot(labels, y, marker="o", label=model)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_ylabel("Population CATE RMSE vs baseline")
    ax.set_title("L: rank-4 size-5 support by treatment Hamming-weight richness")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_support_ladder(metrics: pd.DataFrame, outpath: Path) -> None:
    """Plot H only.  J has its own 16 -> 12 -> 8 -> canonical figure.

    Keeping the H and J scenario lists separate prevents the x/y length mismatch
    that occurred after the J scenarios were inserted into the H order while the H
    labels still contained four entries.
    """
    order = [
        "H_LOCAL_SUPPORT_LADDER_FULL16_DOM",
        "H_LOCAL_SUPPORT_LADDER_R4_REDUNDANT_DOM",
        "H_LOCAL_SUPPORT_LADDER_R4_CANONICAL_DOM",
        "H_LOCAL_SUPPORT_LADDER_R3_REDUNDANT_DOM",
    ]
    labels = [
        "Full 16",
        "rank 4 + redundancy",
        "rank 4 canonical basis",
        "rank 3 + redundancy",
    ]
    assert len(order) == len(labels)

    if "population_cate_rmse_origin" not in metrics.columns:
        return
    present = set(metrics["scenario"].astype(str))
    if any(s not in present for s in order):
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for model, dfm in metrics.groupby("model"):
        y = np.asarray([
            dfm.loc[dfm.scenario == s, "population_cate_rmse_origin"].mean()
            for s in order
        ], dtype=float)
        if np.all(np.isnan(y)):
            continue
        ax.plot(labels, y, marker="o", label=model)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_ylabel("Population CATE RMSE vs baseline")
    ax.set_title("H: localized treatment-support ladder in the dominant X region")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)



def plot_j_rank4_support_smoothing(metrics: pd.DataFrame, outpath: Path) -> None:
    """CATE error while support cardinality shrinks but rank remains exactly four."""
    order = [
        "H_LOCAL_SUPPORT_LADDER_FULL16_DOM",
        "J_RANK4_SUPPORT12_DOM",
        "J_RANK4_SUPPORT8_DOM",
        "H_LOCAL_SUPPORT_LADDER_R4_CANONICAL_DOM",
    ]
    labels = [
        "16 total",
        "12 total",
        "8 total",
        "4 basis + origin",
    ]
    present = set(metrics["scenario"].astype(str))
    if any(s not in present for s in order):
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, dfm in metrics.groupby("model"):
        y = [
            dfm.loc[dfm.scenario == s, "population_cate_rmse_origin"].mean()
            for s in order
        ]
        ax.plot(labels, y, marker="o", label=model)
    ax.set_ylabel("Population CATE RMSE vs baseline")
    ax.set_title("J: rank-4 support smoothing — fewer observed vertices, same span")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_i_theorem_safe(metrics: pd.DataFrame, outpath: Path) -> None:
    """Common-core CATE error for the theorem-safe supported-only ladder."""
    order = [
        "I_THEOREM_SAFE_FULL16",
        "I_THEOREM_SAFE_R4_REDUNDANT",
        "I_THEOREM_SAFE_R4_CANONICAL",
        "I_THEOREM_SAFE_R3_REDUNDANT_SUPPORTED_ONLY",
    ]
    labels = [
        "Full 16",
        "rank 4 + redundancy",
        "rank 4 canonical",
        "rank 3 + redundancy\n(supported only)",
    ]
    if "i_common_cate_rmse_origin" not in metrics.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, dfm in metrics.groupby("model"):
        y = [
            dfm.loc[dfm.scenario == s, "i_common_cate_rmse_origin"].mean()
            for s in order
        ]
        if np.all(np.isnan(np.asarray(y, dtype=float))):
            continue
        ax.plot(labels, y, marker="o", label=model)
    ax.set_ylabel("CATE RMSE on common supported core")
    ax.set_title("I: theorem-safe ladder — no unsupported treatment evaluation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)



def plot_k_neighbor_extrapolation(metrics: pd.DataFrame, outpath: Path) -> None:
    names = [
        "K_NEIGHBOR_FULL16_REFERENCE",
        "K_NEIGHBOR_HALF_CANONICAL",
    ]
    sub = metrics[metrics["scenario"].isin(names)].copy()
    if sub.empty or "k_weak_unsupported_cate_rmse_origin" not in sub.columns:
        return
    agg = (
        sub.groupby(["model", "scenario"], as_index=False)[
            "k_weak_unsupported_cate_rmse_origin"
        ].mean()
    )
    if agg.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(names))
    labels = ["Full16 reference", "Half-X canonical"]
    for model, gm in agg.groupby("model"):
        vals = []
        for name in names:
            s = gm.loc[gm["scenario"] == name, "k_weak_unsupported_cate_rmse_origin"]
            vals.append(float(s.iloc[0]) if len(s) else np.nan)
        ax.plot(x, vals, marker="o", label=model)
    ax.set_xticks(x, labels)
    ax.set_ylabel("CATE RMSE on 11 removed/noncanonical actions")
    ax.set_title("K: neighbor-X treatment extrapolation")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)



def plot_k_unsupported_mod5(metrics: pd.DataFrame, outpath: Path) -> None:
    """CATE extrapolation across the five interleaved support profiles."""
    names = [
        K_UNSUPPORTED_SCENARIO_NAMES["FULL16_REFERENCE"],
        K_UNSUPPORTED_SCENARIO_NAMES["MOD5"],
    ]
    sub = metrics[metrics["scenario"].isin(names)].copy()
    needed = [f"ku_{label}_all16_cate_rmse_origin" for label in ("full","r4","r3","r2","r1")]
    if sub.empty or any(col not in sub.columns for col in needed):
        return
    labels = ["full", "rank 4", "rank 3", "rank 2", "rank 1"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(5)
    for model, dfm in sub.groupby("model"):
        for scenario_name, line_label in [
            (K_UNSUPPORTED_SCENARIO_NAMES["FULL16_REFERENCE"], f"{model} reference"),
            (K_UNSUPPORTED_SCENARIO_NAMES["MOD5"], f"{model} mod-5"),
        ]:
            d = dfm[dfm["scenario"] == scenario_name]
            if d.empty:
                continue
            vals = [float(d[col].mean()) for col in needed]
            ax.plot(x, vals, marker="o", label=line_label)
    ax.set_xticks(x, labels)
    ax.set_ylabel("CATE RMSE evaluated over all 16 treatments")
    ax.set_title("K-UNSUPPORTED: interleaved X support ladder")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)



def plot_k_random_mod8(metrics: pd.DataFrame, outpath: Path) -> None:
    """Overall and modulo-specific CATE for the diverse rank-4 K-RANDOM design."""
    names = [
        K_RANDOM_SCENARIO_NAMES["FULL16_REFERENCE"],
        K_RANDOM_SCENARIO_NAMES["MOD8"],
    ]
    sub = metrics[metrics["scenario"].isin(names)].copy()
    if sub.empty or "kr_overall_all16_cate_rmse_origin" not in sub.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(8)
    for model, dfm in sub.groupby("model"):
        for scenario_name, line_label in [
            (K_RANDOM_SCENARIO_NAMES["FULL16_REFERENCE"], f"{model} reference"),
            (K_RANDOM_SCENARIO_NAMES["MOD8"], f"{model} diverse rank-4 mod-8"),
        ]:
            d = dfm[dfm["scenario"] == scenario_name]
            if d.empty:
                continue
            vals = []
            for mod in range(8):
                col = f"kr_m{mod}_all16_cate_rmse_origin"
                vals.append(float(d[col].mean()) if col in d.columns else np.nan)
            ax.plot(x, vals, marker="o", label=line_label)
    ax.set_xticks(x, [f"M{m}" for m in range(8)])
    ax.set_xlabel("Ordered-X modulo-8 support class")
    ax.set_ylabel("CATE RMSE evaluated over all 16 treatments")
    ax.set_title("K-RANDOM: eight interleaved diverse rank-4 supports")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)



def plot_k_random3_mod8(metrics: pd.DataFrame, outpath: Path) -> None:
    """Modulo-specific all-16 CATE for alternating rank-3/rank-2 K-RANDOM3."""
    names = [
        K_RANDOM_SCENARIO_NAMES["FULL16_REFERENCE"],
        K_RANDOM3_SCENARIO_NAMES["MOD8"],
    ]
    sub = metrics[metrics["scenario"].isin(names)].copy()
    if sub.empty or "kr3_overall_all16_cate_rmse_origin" not in sub.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(8)
    xticklabels = [f"M{m} r{3 if m % 2 == 0 else 2}" for m in range(8)]
    for model, dfm in sub.groupby("model"):
        for scenario_name, line_label in [
            (K_RANDOM_SCENARIO_NAMES["FULL16_REFERENCE"], f"{model} full16 reference"),
            (K_RANDOM3_SCENARIO_NAMES["MOD8"], f"{model} rank3/rank2 mod-8"),
        ]:
            d = dfm[dfm["scenario"] == scenario_name]
            if d.empty:
                continue
            vals = []
            for mod in range(8):
                col = f"kr3_m{mod}_all16_cate_rmse_origin"
                vals.append(float(d[col].mean()) if col in d.columns else np.nan)
            ax.plot(x, vals, marker="o", label=line_label)
    ax.set_xticks(x, xticklabels)
    ax.set_xlabel("Ordered-X modulo-8 support class")
    ax.set_ylabel("CATE RMSE evaluated over all 16 treatments")
    ax.set_title("K-RANDOM3: interleaved rank-3/rank-2 supports, no local rank 4")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)



def plot_k_extreme(metrics: pd.DataFrame, outpath: Path) -> None:
    """Compare ordinary and maximum-module K-RANDOM designs on all-16 CATE."""
    order = [
        K_RANDOM_SCENARIO_NAMES["FULL16_REFERENCE"],
        K_RANDOM_SCENARIO_NAMES["MOD8"],
        K_EXTREME_SCENARIO_NAMES["RANK4"],
        K_RANDOM3_SCENARIO_NAMES["MOD8"],
        K_EXTREME_SCENARIO_NAMES["LT4"],
    ]
    labels = ["Full16", "R4 mod8", "R4 maxmod", "R3/R2 mod8", "rank<4 maxmod"]
    sub = metrics[metrics["scenario"].isin(order)].copy()
    metric = "kext_overall_all16_cate_rmse_origin"
    if sub.empty or metric not in sub.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(order))
    for model, dfm in sub.groupby("model"):
        vals = []
        for scenario_name in order:
            d = dfm[dfm["scenario"] == scenario_name]
            vals.append(float(d[metric].mean()) if not d.empty else np.nan)
        ax.plot(x, vals, marker="o", label=model)
    ax.set_xticks(x, labels)
    ax.set_ylabel("CATE RMSE evaluated over all 16 treatments")
    ax.set_title("K-EXTREME: maximum local support-orientation diversity")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_treatment_recovery(metrics: pd.DataFrame, outpath: Path) -> None:
    """Exact optimal-treatment recovery across scenarios."""
    preferred_order = [
        FULL2K_BASELINE_NAME,
        "A_STRONG",
        "B_WEAK",
        "C_RANK3",
        "D_REF_X",
        "D_POOR_X",
        "G_GENERAL_REDUNDANT_RANK4_DOM",
        "G_GENERAL_REDUNDANT_RANK3_DOM",
        "G_GENERAL_REDUNDANT_RANK2_DOM",
        "G_GENERAL_REDUNDANT_RANK1_DOM",
        "H_LOCAL_SUPPORT_LADDER_FULL16_DOM",
        "H_LOCAL_SUPPORT_LADDER_R4_REDUNDANT_DOM",
        "H_LOCAL_SUPPORT_LADDER_R4_CANONICAL_DOM",
        "H_LOCAL_SUPPORT_LADDER_R3_REDUNDANT_DOM",
    ]
    present = set(metrics["scenario"].astype(str))
    order = [s for s in preferred_order if s in present]
    if not order:
        return
    labels = [
        s.replace("BASE_FULL_2K_GIVEN_X", "Base")
         .replace("G_GENERAL_REDUNDANT_", "G-")
         .replace("H_LOCAL_SUPPORT_LADDER_", "H-")
         .replace("J_RANK4_", "J-")
         .replace("_DOM", "")
        for s in order
    ]
    fig, ax = plt.subplots(figsize=(13, 5))
    for model, dfm in metrics.groupby("model"):
        y = [
            dfm.loc[dfm.scenario == s, "population_treatment_recovery_accuracy"].mean()
            for s in order
        ]
        ax.plot(labels, y, marker="o", label=model)
    ax.set_ylabel("Exact treatment recovery accuracy")
    ax.set_title("Optimal intervention recovery")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_relative_worsening(metrics: pd.DataFrame, outpath: Path) -> None:
    """Primary percent deterioration in population CATE RMSE versus full 2^K|x."""
    col = "pct_worse_vs_full2k__population_cate_rmse_origin"
    if col not in metrics.columns:
        return
    order = [
        "A_STRONG", "B_WEAK", "C_RANK3",
        "D_REF_X", "D_POOR_X",
    ]
    labels = ["A", "B", "C", "D ref X", "D poor X"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, dfm in metrics.groupby("model"):
        y = [dfm.loc[dfm.scenario == s, col].mean() for s in order]
        ax.plot(labels, y, marker="o", label=model)
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel("% worsening in population CATE RMSE vs full 2^K|x")
    ax.set_title("Relative deterioration from literal full-support baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


# =====================================================================
# Experiment runner
# =====================================================================

def scenario_catalog_rows(scenarios: Mapping[str, Scenario], rank_map: Mapping[str, int] | None = None) -> List[Dict[str, object]]:
    rows = []
    for s in scenarios.values():
        rows.append({
            "scenario": s.name,
            "family": s.family,
            "description": s.description,
            "target_kind": s.target_kind,
            "dominant_rank": s.metadata.get("dominant_rank", np.nan),
            "dominant_region": s.metadata.get("dominant_region", ""),
            "dominant_profile": s.metadata.get("dominant_profile", ""),
            "dominant_n_support_vertices": s.metadata.get("dominant_n_support_vertices", np.nan),
            "dominant_share": s.metadata.get("dominant_share", np.nan),
            "missing_component_index": s.metadata.get("missing_component_index", np.nan),
            "omitted_direction_code": s.metadata.get("omitted_direction_code", np.nan),
            "omitted_direction_bits": s.metadata.get("omitted_direction_bits", ""),
            "redundant_vertex_code": s.metadata.get("redundant_vertex_code", np.nan),
            "redundant_vertex_bits": s.metadata.get("redundant_vertex_bits", ""),
            "retained_basis_bits": str(s.metadata.get("retained_basis_bits", "")),
            "evaluation_mode": s.metadata.get("evaluation_mode", ""),
            "evaluation_support_bits": str(s.metadata.get("evaluation_support_bits", "")),
            "common_evaluation_bits": str(s.metadata.get("common_evaluation_bits", "")),
            "support_rank": s.metadata.get("support_rank", np.nan),
            "support_cardinality": s.metadata.get("support_cardinality", np.nan),
            "j_total_support_vertices": s.metadata.get("j_total_support_vertices", np.nan),
            "j_nonzero_support_vertices": s.metadata.get("j_nonzero_support_vertices", np.nan),
            "j_extra_joint_vertices": s.metadata.get("j_extra_joint_vertices", np.nan),
            "j_joint_vertices_bits": str(s.metadata.get("j_joint_vertices_bits", "")),
            "k_profile": s.metadata.get("k_profile", ""),
            "k_n_weak_x": s.metadata.get("k_n_weak_x", np.nan),
            "k_n_full_x": s.metadata.get("k_n_full_x", np.nan),
            "k_weak_support_bits": str(s.metadata.get("k_weak_support_bits", "")),
            "k_full_support_bits": str(s.metadata.get("k_full_support_bits", "")),
            "k_order_direction": s.metadata.get("k_order_direction", ""),
            "k_neighbor_s_extra_mean": s.metadata.get("k_neighbor_s_extra_mean", np.nan),
            "k_neighbor_xstar_l2_mean": s.metadata.get("k_neighbor_xstar_l2_mean", np.nan),
            "ku_profile": s.metadata.get("ku_profile", ""),
            "ku_order_direction": s.metadata.get("ku_order_direction", ""),
            "ku_group_sizes": str(s.metadata.get("ku_group_sizes", "")),
            "ku_support_bits": str(s.metadata.get("ku_support_bits", "")),
            "ku_rank_by_group": str(s.metadata.get("ku_rank_by_group", "")),
            "ku_neighbor_s_extra_mean": s.metadata.get("ku_neighbor_s_extra_mean", np.nan),
            "ku_neighbor_xstar_l2_mean": s.metadata.get("ku_neighbor_xstar_l2_mean", np.nan),
            "kr_profile": s.metadata.get("kr_profile", ""),
            "kr_order_direction": s.metadata.get("kr_order_direction", ""),
            "kr_group_sizes": str(s.metadata.get("kr_group_sizes", "")),
            "kr_support_bits": str(s.metadata.get("kr_support_bits", "")),
            "kr_support_permutation": str(s.metadata.get("kr_support_permutation", "")),
            "kr_nonzero_usage": str(s.metadata.get("kr_nonzero_usage", "")),
            "kr_max_pairwise_nonzero_overlap": s.metadata.get("kr_max_pairwise_nonzero_overlap", np.nan),
            "kr_internal_hamming_diversity_min": s.metadata.get("kr_internal_hamming_diversity_min", np.nan),
            "kr_neighbor_s_extra_mean": s.metadata.get("kr_neighbor_s_extra_mean", np.nan),
            "kr_neighbor_xstar_l2_mean": s.metadata.get("kr_neighbor_xstar_l2_mean", np.nan),
            "kr3_profile": s.metadata.get("kr3_profile", ""),
            "kr3_evaluation_mode": s.metadata.get("kr3_evaluation_mode", ""),
            "kr3_order_direction": s.metadata.get("kr3_order_direction", ""),
            "kr3_group_sizes": str(s.metadata.get("kr3_group_sizes", "")),
            "kr3_support_bits": str(s.metadata.get("kr3_support_bits", "")),
            "kr3_rank_by_mod": str(s.metadata.get("kr3_rank_by_mod", "")),
            "kr3_cardinality_by_mod": str(s.metadata.get("kr3_cardinality_by_mod", "")),
            "kr3_nonzero_usage": str(s.metadata.get("kr3_nonzero_usage", "")),
            "kr3_total_hamming_diversity": s.metadata.get("kr3_total_hamming_diversity", np.nan),
            "kr3_max_pairwise_nonzero_overlap": s.metadata.get("kr3_max_pairwise_nonzero_overlap", np.nan),
            "kr3_neighbor_s_extra_mean": s.metadata.get("kr3_neighbor_s_extra_mean", np.nan),
            "kr3_neighbor_xstar_l2_mean": s.metadata.get("kr3_neighbor_xstar_l2_mean", np.nan),
            "kext_variant": s.metadata.get("kext_variant", ""),
            "kext_theoretical_module": s.metadata.get("kext_theoretical_module", np.nan),
            "kext_effective_module": s.metadata.get("kext_effective_module", np.nan),
            "kext_n_unique_supports_used": s.metadata.get("kext_n_unique_supports_used", np.nan),
            "kext_rank_counts": str(s.metadata.get("kext_rank_counts", "")),
            "kext_cardinality_counts": str(s.metadata.get("kext_cardinality_counts", "")),
            "l_dominant_profile": s.metadata.get("l_dominant_profile", ""),
            "l_dominant_region": s.metadata.get("l_dominant_region", ""),
            "l_dominant_share": s.metadata.get("l_dominant_share", np.nan),
            "l_profile_map": str(s.metadata.get("l_profile_map", "")),
            "l_weight_patterns": str(s.metadata.get("l_weight_patterns", "")),
            "l_region_support_bits": str(s.metadata.get("l_region_support_bits", "")),
            "l_region_basis_bits": str(s.metadata.get("l_region_basis_bits", "")),
            "l_anchor_allocation": str(s.metadata.get("l_anchor_allocation", "")),
            "l_n_anchors": s.metadata.get("l_n_anchors", np.nan),
            "l_n_train_rows": s.metadata.get("l_n_train_rows", np.nan),
            "l_controlled_crossed_support": int(bool(s.metadata.get("l_controlled_crossed_support", False))),
            "l_all16_evaluation": int(bool(s.metadata.get("l_all16_evaluation", False))),
            "literal_all_2k_per_x": int(s.metadata.get("literal_all_2k_per_x", 0)),
            "n_unique_train_x": s.n_unique_training_x(),
            "n_train_rows": s.n_training_rows(),
            "rank_map": str(dict(rank_map)) if rank_map is not None and s.family.startswith(("E_", "F_", "G_")) else "",
        })
    return rows



def postprocess_existing_outputs(outdir: Path) -> None:
    """Regenerate summaries/figures from an already completed simulation.

    This is useful when model fitting completed successfully but a downstream
    plotting/export step failed.  It deliberately does not refit any model.
    """
    metrics_path = outdir / "seed_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Cannot postprocess: {metrics_path} does not exist."
        )

    print(f"Post-processing existing metrics from {metrics_path.resolve()}", flush=True)
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        raise RuntimeError("seed_metrics.csv is empty; nothing to postprocess.")

    # Rebuild summary with the de-fragmented aggregation path.
    summarize_metrics(metrics).to_csv(outdir / "summary.csv", index=False)

    # Re-export family-specific metric tables from the preserved seed metrics.
    if "family" in metrics.columns:
        metrics[metrics["family"] == "I_THEOREM_SAFE_SUPPORT_LADDER"].to_csv(
            outdir / "i_theorem_safe_supported_only_metrics.csv", index=False
        )
        metrics[metrics["family"] == "L_RANK4_NORM_RICHNESS"].to_csv(
            outdir / "l_rank4_norm_richness_metrics.csv", index=False
        )
        metrics[metrics["family"] == "K_NEIGHBOR_SUPPORT_EXTRAPOLATION"].to_csv(
            outdir / "k_neighbor_extrapolation_metrics.csv", index=False
        )
        metrics[metrics["family"] == "K_UNSUPPORTED_MOD5_EXTRAPOLATION"].to_csv(
            outdir / "k_unsupported_mod5_extrapolation_metrics.csv", index=False
        )
        metrics[metrics["family"] == "K_RANDOM_MOD8_RANK4_EXTRAPOLATION"].to_csv(
            outdir / "k_random_mod8_rank4_extrapolation_metrics.csv", index=False
        )
        # K-RANDOM3 reuses the K-RANDOM full16 reference.  Include both rows so
        # the exported table is immediately matched for analysis.
        kr3_names = [K_RANDOM_SCENARIO_NAMES["FULL16_REFERENCE"], K_RANDOM3_SCENARIO_NAMES["MOD8"]]
        metrics[metrics["scenario"].isin(kr3_names)].to_csv(
            outdir / "k_random3_mod8_rank32_extrapolation_metrics.csv", index=False
        )
        kext_names = [
            K_RANDOM_SCENARIO_NAMES["FULL16_REFERENCE"],
            K_EXTREME_SCENARIO_NAMES["RANK4"],
            K_EXTREME_SCENARIO_NAMES["LT4"],
        ]
        metrics[metrics["scenario"].isin(kext_names)].to_csv(
            outdir / "k_extreme_maxmod_extrapolation_metrics.csv", index=False
        )

    j_path_names = [
        "H_LOCAL_SUPPORT_LADDER_FULL16_DOM",
        "J_RANK4_SUPPORT12_DOM",
        "J_RANK4_SUPPORT8_DOM",
        "H_LOCAL_SUPPORT_LADDER_R4_CANONICAL_DOM",
    ]
    metrics[metrics["scenario"].isin(j_path_names)].to_csv(
        outdir / "j_rank4_support_smoothing_metrics.csv", index=False
    )

    # Recreate every figure.  Individual plotters are intentionally defensive
    # about absent scenarios/columns so this also works with older 32-scenario
    # runs that predate K-RANDOM.
    plot_core(metrics, outdir / "core_support_validation.png")
    plot_relative_worsening(metrics, outdir / "relative_worsening_vs_full2k.png")
    plot_regional(metrics, "E_CANONICAL", outdir / "regional_canonical_cate.png")
    plot_regional(metrics, "F_GENERAL_BASIS", outdir / "regional_general_basis_cate.png")
    plot_regional(metrics, "G_GENERAL_REDUNDANT", outdir / "regional_general_redundant_cate.png")
    plot_l_rank4_norm_richness(metrics, outdir / "l_rank4_norm_richness_cate.png")
    plot_support_ladder(metrics, outdir / "h_local_support_ladder_cate.png")
    plot_j_rank4_support_smoothing(metrics, outdir / "j_rank4_support_smoothing_cate.png")
    plot_i_theorem_safe(metrics, outdir / "i_theorem_safe_common_core_cate.png")
    plot_k_neighbor_extrapolation(metrics, outdir / "k_neighbor_extrapolation_cate.png")
    plot_k_unsupported_mod5(metrics, outdir / "k_unsupported_mod5_cate.png")
    plot_k_random_mod8(metrics, outdir / "k_random_mod8_rank4_cate.png")
    plot_k_random3_mod8(metrics, outdir / "k_random3_mod8_rank32_cate.png")
    plot_k_extreme(metrics, outdir / "k_extreme_maxmod_cate.png")
    plot_treatment_recovery(metrics, outdir / "treatment_recovery_all_scenarios.png")

    print("Post-processing completed without refitting models.", flush=True)

def run_experiment(
    replicates: int,
    outdir: Path,
    max_train_n: int | None = MAX_TRAIN_N_DEFAULT,
    dominant_share: float = DOMINANT_SHARE_DEFAULT,
    training_outcome: str = "clean",
    max_attempt_multiplier: int = 12,
    model_config: Mapping[str, Any] | None = None,
    selected_models: Sequence[str] = AVAILABLE_MODELS,
    selected_families: Sequence[str] = tuple(
        name for name in AVAILABLE_FAMILIES if name != "K"
    ),
    device: str = "cpu",
    base_seed: int = BASE_SEED,
    sample_n: int = N,
    eval_fraction: float = POP_EVAL_FRACTION,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    selected_families = expand_family_dependencies(selected_families)
    resolved_model_config = dict(model_config or {})
    factories = build_model_factories(resolved_model_config, selected_models, device)
    if not factories:
        raise ValueError("No model factory was selected.")
    bootstrap_repetitions = int(
        resolved_model_config.get("bootstrap_repetitions", 5000)
    )
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be at least 1.")

    metric_rows: List[Dict[str, object]] = []
    support_rows: List[Dict[str, object]] = []
    ate_rows: List[Dict[str, object]] = []
    cate_rows: List[Dict[str, object]] = []
    policy_rows: List[Dict[str, object]] = []
    catalog_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []

    successful = 0
    attempt = 0
    max_attempts = max(replicates * max_attempt_multiplier, replicates)

    while successful < replicates and attempt < max_attempts:
        seed = base_seed + attempt
        attempt += 1
        rotation = successful % 4
        seed_started = time.perf_counter()
        print(
            f"\n=== Preparing repetition {successful + 1}/{replicates} | "
            f"seed={seed} | attempt={attempt}/{max_attempts} ===",
            flush=True,
        )

        try:
            data = generate_base_dgp(seed, n=sample_n)
            region_labels, cuts = define_x_regions(data)
            train_pool, population_eval_idx = common_population_eval_split(
                region_labels, seed + 40_000, eval_fraction=eval_fraction
            )

            scenarios, scenario_metadata = build_selected_family_scenarios(
                data=data,
                train_pool=train_pool,
                population_eval_idx=population_eval_idx,
                region_labels=region_labels,
                rotation=rotation,
                seed=seed,
                dominant_share=dominant_share,
                max_train_n=max_train_n,
                selected_families=selected_families,
                cycle_index=successful,
            )

            # Verify formal local geometry before fitting any model.
            for s in scenarios.values():
                diag_rows = scenario_support_diagnostics_rows(data, s, region_labels)
                support_rows.extend(diag_rows)

                if s.name == FULL2K_BASELINE_NAME:
                    for row in diag_rows:
                        if (
                            row["all_2k_per_x"] != 1
                            or row["all_16_vertices_present"] != 1
                            or row["rank_U"] != K_BITS
                        ):
                            raise RuntimeError(
                                "Literal full-2^K|x baseline failed exact per-x support."
                            )
                if s.name == "B_WEAK":
                    for row in diag_rows:
                        if row["origin_present"] != 1 or row["rank_U"] != K_BITS:
                            raise RuntimeError("B_WEAK failed baseline/full-rank local support.")
                if s.name == "C_RANK3":
                    for row in diag_rows:
                        if row["origin_present"] != 1 or row["rank_U"] != 3:
                            raise RuntimeError("C_RANK3 failed intended local rank=3 support.")
                if s.family == "L_RANK4_NORM_RICHNESS":
                    profile_map = {
                        str(k): str(v) for k, v in s.metadata["l_profile_map"].items()
                    }
                    if set(profile_map.values()) != set(L_PROFILE_ORDER):
                        raise RuntimeError(f"{s.name}: L profiles are not one-to-one across X regions.")
                    for region in X_REGIONS:
                        support = tuple(sorted(map(int, s.local_supports[region])))
                        basis = tuple(sorted(map(int, s.local_bases[region])))
                        profile = profile_map[region]
                        if len(support) != K_BITS + 1 or rank_of_codes(support) != K_BITS:
                            raise RuntimeError(
                                f"{s.name}: L {region}/{profile} must be rank4 and size5."
                            )
                        if support != tuple(sorted((ORIGIN, *basis))):
                            raise RuntimeError(f"{s.name}: L support/basis mismatch in {region}.")
                        observed_pattern = tuple(
                            sorted(treatment_hamming_weight(code) for code in basis)
                        )
                        if observed_pattern != L_WEIGHT_PATTERNS[profile]:
                            raise RuntimeError(
                                f"{s.name}: L {region}/{profile} has weights "
                                f"{observed_pattern}, expected {L_WEIGHT_PATTERNS[profile]}."
                            )
                    dom_profile = str(s.metadata["l_dominant_profile"])
                    dom_region = str(s.metadata["l_dominant_region"])
                    if profile_map[dom_region] != dom_profile:
                        raise RuntimeError(f"{s.name}: L dominant profile/region mismatch.")
                    if not bool(s.metadata.get("l_all16_evaluation", False)):
                        raise RuntimeError(f"{s.name}: L must evaluate all 16 treatments.")
                    src_l = s.training_source_idx()
                    t_l = s.training_t(data)
                    for x in np.unique(src_l):
                        region = str(region_labels[int(x)])
                        observed = tuple(
                            sorted(np.unique(t_l[src_l == x]).astype(int).tolist())
                        )
                        expected = tuple(sorted(map(int, s.local_supports[region])))
                        if observed != expected or int(np.sum(src_l == x)) != K_BITS + 1:
                            raise RuntimeError(
                                f"{s.name}: L anchor x={int(x)} does not have exactly "
                                f"the intended five-vertex support for {region}."
                            )
                if s.family == "H_LOCAL_SUPPORT_LADDER":
                    dom = str(s.metadata["dominant_region"])
                    profile = str(s.metadata["dominant_profile"])
                    support = tuple(s.local_supports[dom])
                    geom = geometry_diagnostics(support)
                    expected = {
                        "FULL16": (4, 16),
                        "R4_REDUNDANT": (4, 6),
                        "R4_CANONICAL": (4, 5),
                        "R3_REDUNDANT": (3, 5),
                    }[profile]
                    if (int(geom["rank_U"]), len(support)) != expected:
                        raise RuntimeError(
                            f"{s.name} failed H ladder geometry: "
                            f"got rank={geom['rank_U']}, |S|={len(support)}, expected={expected}."
                        )
                if s.family == "J_RANK4_SUPPORT_SMOOTHING":
                    expected_size = int(s.metadata["j_total_support_vertices"])
                    dom = str(s.metadata["dominant_region"])
                    support = tuple(s.local_supports[dom])
                    if rank_of_codes(support) != K_BITS or len(support) != expected_size:
                        raise RuntimeError(
                            f"{s.name} failed J full-rank smoothing geometry: "
                            f"rank={rank_of_codes(support)}, |S|={len(support)}, "
                            f"expected rank=4, |S|={expected_size}."
                        )
                    canonical = set((ORIGIN, *CANONICAL_NONZERO))
                    if not canonical.issubset(set(support)):
                        raise RuntimeError(f"{s.name}: J support lost a canonical direction.")
                    for region in X_REGIONS:
                        if region == dom:
                            continue
                        if tuple(sorted(s.local_supports[region])) != tuple(range(N_REGIMES)):
                            raise RuntimeError(f"{s.name}: non-dominant J region lost full support.")
                if s.family == "K_NEIGHBOR_SUPPORT_EXTRAPOLATION":
                    profile = str(s.metadata["k_profile"])
                    src_k = s.training_source_idx()
                    t_k = s.training_t(data)
                    weak_k = set(map(int, s.metadata["k_weak_anchor_idx"]))
                    full_k = set(map(int, s.metadata["k_full_anchor_idx"]))
                    canonical_k = tuple(sorted((ORIGIN, *CANONICAL_NONZERO)))
                    all_k = tuple(range(N_REGIMES))
                    if not weak_k or not full_k or len(weak_k) != len(full_k):
                        raise RuntimeError(f"{s.name}: invalid K weak/full X split.")
                    for region in X_REGIONS:
                        wreg = [x for x in weak_k if region_labels[x] == region]
                        freg = [x for x in full_k if region_labels[x] == region]
                        if len(wreg) == 0 or len(wreg) != len(freg):
                            raise RuntimeError(f"{s.name}: K split is not balanced inside {region}.")
                    for x in weak_k:
                        obs = tuple(sorted(np.unique(t_k[src_k == x]).astype(int).tolist()))
                        expected = all_k if profile == "FULL16_REFERENCE" else canonical_k
                        if obs != expected:
                            raise RuntimeError(f"{s.name}: incorrect K weak-anchor support at x={x}.")
                    for x in full_k:
                        obs = tuple(sorted(np.unique(t_k[src_k == x]).astype(int).tolist()))
                        if obs != all_k:
                            raise RuntimeError(f"{s.name}: K full neighbor lost a treatment at x={x}.")
                if s.family == "K_UNSUPPORTED_MOD5_EXTRAPOLATION":
                    profile = str(s.metadata["ku_profile"])
                    src_ku = s.training_source_idx()
                    t_ku = s.training_t(data)
                    groups_ku = {
                        str(k): set(map(int, v))
                        for k, v in s.metadata["ku_group_anchor_idx"].items()
                    }
                    supports_ku = {
                        str(k): tuple(sorted(map(int, v)))
                        for k, v in s.metadata["ku_support_codes"].items()
                    }
                    union = set().union(*groups_ku.values())
                    if len(union) != sum(len(v) for v in groups_ku.values()):
                        raise RuntimeError(f"{s.name}: K-UNSUPPORTED groups overlap.")
                    for region in X_REGIONS:
                        sizes = [sum(region_labels[x] == region for x in groups_ku[label]) for label in ("full","r4","r3","r2","r1")]
                        if min(sizes) <= 0 or len(set(sizes)) != 1:
                            raise RuntimeError(f"{s.name}: K-UNSUPPORTED modulo groups are not balanced in {region}.")
                    for label in ("full", "r4", "r3", "r2", "r1"):
                        expected_mixed = supports_ku[label]
                        for x in groups_ku[label]:
                            obs = tuple(sorted(np.unique(t_ku[src_ku == x]).astype(int).tolist()))
                            expected = tuple(range(N_REGIMES)) if profile == "FULL16_REFERENCE" else expected_mixed
                            if obs != expected:
                                raise RuntimeError(
                                    f"{s.name}: K-UNSUPPORTED support mismatch for group {label} at x={x}."
                                )
                            expected_rank = K_BITS if profile == "FULL16_REFERENCE" else int(s.metadata["ku_rank_by_group"][label])
                            if rank_of_codes(obs) != expected_rank:
                                raise RuntimeError(
                                    f"{s.name}: K-UNSUPPORTED rank mismatch for group {label} at x={x}."
                                )
                if s.family == "K_RANDOM_MOD8_RANK4_EXTRAPOLATION":
                    profile = str(s.metadata["kr_profile"])
                    src_kr = s.training_source_idx()
                    t_kr = s.training_t(data)
                    groups_kr = {
                        int(k): set(map(int, v))
                        for k, v in s.metadata["kr_group_anchor_idx"].items()
                    }
                    supports_kr = {
                        int(k): tuple(sorted(map(int, v)))
                        for k, v in s.metadata["kr_support_codes"].items()
                    }
                    union = set().union(*groups_kr.values())
                    if len(union) != sum(len(v) for v in groups_kr.values()):
                        raise RuntimeError(f"{s.name}: K-RANDOM groups overlap.")
                    if tuple(sorted(set().union(*(set(v) for v in supports_kr.values())))) != tuple(range(N_REGIMES)):
                        raise RuntimeError(f"{s.name}: K-RANDOM support union is not full16.")
                    for region in X_REGIONS:
                        sizes = [sum(region_labels[x] == region for x in groups_kr[m]) for m in range(8)]
                        if min(sizes) <= 0 or len(set(sizes)) != 1:
                            raise RuntimeError(f"{s.name}: K-RANDOM modulo groups are not balanced in {region}.")
                    for mod in range(8):
                        expected_mixed = supports_kr[mod]
                        if len(expected_mixed) != 5 or rank_of_codes(expected_mixed) != K_BITS:
                            raise RuntimeError(f"{s.name}: K-RANDOM mod {mod} is not rank4/size5.")
                        for x in groups_kr[mod]:
                            obs = tuple(sorted(np.unique(t_kr[src_kr == x]).astype(int).tolist()))
                            expected = tuple(range(N_REGIMES)) if profile == "FULL16_REFERENCE" else expected_mixed
                            if obs != expected:
                                raise RuntimeError(
                                    f"{s.name}: K-RANDOM support mismatch for mod {mod} at x={x}."
                                )
                if "kr3_support_codes" in s.metadata:
                    profile3 = str(s.metadata.get("kr3_profile", ""))
                    src_kr3 = s.training_source_idx()
                    t_kr3 = s.training_t(data)
                    groups_kr3 = {
                        int(k): set(map(int, v))
                        for k, v in s.metadata["kr3_group_anchor_idx"].items()
                    }
                    supports_kr3 = {
                        int(k): tuple(sorted(map(int, v)))
                        for k, v in s.metadata["kr3_support_codes"].items()
                    }
                    ranks_kr3 = {int(k): int(v) for k, v in s.metadata["kr3_rank_by_mod"].items()}
                    cards_kr3 = {int(k): int(v) for k, v in s.metadata["kr3_cardinality_by_mod"].items()}
                    if set(groups_kr3) != set(range(8)) or set(supports_kr3) != set(range(8)):
                        raise RuntimeError(f"{s.name}: K-RANDOM3 must define modulo classes 0..7.")
                    if tuple(sorted(set().union(*(set(v) for v in supports_kr3.values())))) != tuple(range(N_REGIMES)):
                        raise RuntimeError(f"{s.name}: K-RANDOM3 support union is not full16.")
                    if any(ranks_kr3[m] not in (2, 3) for m in range(8)):
                        raise RuntimeError(f"{s.name}: K-RANDOM3 metadata contains a rank outside {{2,3}}.")
                    if any(ranks_kr3[m] == K_BITS for m in range(8)):
                        raise RuntimeError(f"{s.name}: K-RANDOM3 must never contain local rank4 support.")
                    for mod in range(8):
                        expected_rank = 3 if mod % 2 == 0 else 2
                        expected_card = 4 if mod % 2 == 0 else 3
                        if ranks_kr3[mod] != expected_rank or cards_kr3[mod] != expected_card:
                            raise RuntimeError(
                                f"{s.name}: K-RANDOM3 mod {mod} expected rank/card "
                                f"{expected_rank}/{expected_card}, got {ranks_kr3[mod]}/{cards_kr3[mod]}."
                            )
                        support = supports_kr3[mod]
                        if rank_of_codes(support) != expected_rank or len(support) != expected_card:
                            raise RuntimeError(f"{s.name}: K-RANDOM3 support geometry mismatch at mod {mod}.")
                        for x in groups_kr3[mod]:
                            obs = tuple(sorted(np.unique(t_kr3[src_kr3 == x]).astype(int).tolist()))
                            expected_obs = tuple(range(N_REGIMES)) if profile3 == "FULL16_REFERENCE" else support
                            if obs != expected_obs:
                                raise RuntimeError(
                                    f"{s.name}: K-RANDOM3 support mismatch for mod {mod} at x={x}."
                                )
                    if profile3 == "MOD8" and any(
                        rank_of_codes(tuple(sorted(np.unique(t_kr3[src_kr3 == x]).astype(int).tolist()))) >= K_BITS
                        for x in set().union(*groups_kr3.values())
                    ):
                        raise RuntimeError(f"{s.name}: K-RANDOM3 observed rank4 at a mixed-training X anchor.")
                if s.family == "K_EXTREME_MAXMOD_EXTRAPOLATION":
                    variant = str(s.metadata["kext_variant"])
                    src_ke = s.training_source_idx()
                    t_ke = s.training_t(data)
                    anchors_ke = set(map(int, s.metadata["kext_selected_anchor_idx"]))
                    support_map_ke = {
                        int(k): tuple(sorted(map(int, v)))
                        for k, v in s.metadata["kext_anchor_support_codes"].items()
                    }
                    if set(support_map_ke) != anchors_ke:
                        raise RuntimeError(f"{s.name}: K-EXTREME support map does not cover the exact anchors.")
                    obs_ke: Dict[int, List[int]] = {}
                    for x, t in zip(src_ke, t_ke):
                        obs_ke.setdefault(int(x), []).append(int(t))
                    obs_ke2 = {x: tuple(sorted(set(v))) for x, v in obs_ke.items()}
                    if set(obs_ke2) != anchors_ke:
                        raise RuntimeError(f"{s.name}: K-EXTREME training rows lost X anchors.")
                    ranks_seen = []
                    for x in anchors_ke:
                        expected = support_map_ke[x]
                        observed = obs_ke2[x]
                        if observed != expected:
                            raise RuntimeError(f"{s.name}: K-EXTREME support mismatch at x={x}.")
                        rr = rank_of_codes(observed)
                        ranks_seen.append(rr)
                        if variant == "RANK4_MAXMOD":
                            if rr != 4 or len(observed) != 5:
                                raise RuntimeError(f"{s.name}: rank4 extreme support is not rank4/size5 at x={x}.")
                        elif variant == "LT4_MAXMOD":
                            if rr not in (1, 2, 3) or len(observed) != rr + 1:
                                raise RuntimeError(f"{s.name}: rank<4 extreme support is invalid at x={x}.")
                        else:
                            raise RuntimeError(f"{s.name}: unknown K-EXTREME variant {variant}.")
                    n_unique = len(set(support_map_ke.values()))
                    if n_unique != int(s.metadata["kext_effective_module"]):
                        raise RuntimeError(f"{s.name}: K-EXTREME effective module mismatch.")
                    if n_unique != int(s.metadata["kext_n_unique_supports_used"]):
                        raise RuntimeError(f"{s.name}: K-EXTREME unique-support count mismatch.")
                    if tuple(sorted(set().union(*(set(v) for v in support_map_ke.values())))) != tuple(range(N_REGIMES)):
                        raise RuntimeError(f"{s.name}: K-EXTREME empirical support union is not full16.")
                    if variant == "RANK4_MAXMOD" and any(r != 4 for r in ranks_seen):
                        raise RuntimeError(f"{s.name}: K-EXTREME rank4 observed a non-rank4 local support.")
                    if variant == "LT4_MAXMOD" and any(r >= 4 for r in ranks_seen):
                        raise RuntimeError(f"{s.name}: K-EXTREME rank<4 observed local rank4.")
                if s.family == "I_THEOREM_SAFE_SUPPORT_LADDER":
                    profile = str(s.metadata["i_profile"])
                    expected = {
                        "FULL16": (4, 16),
                        "R4_REDUNDANT": (4, 6),
                        "R4_CANONICAL": (4, 5),
                        "R3_REDUNDANT": (3, 5),
                    }[profile]
                    eval_codes = set(map(int, s.metadata["evaluation_support_codes"]))
                    common_codes = set(map(int, s.metadata["common_evaluation_codes"]))
                    for row in diag_rows:
                        observed = set(map(int, s.local_supports[row["x_region"]]))
                        if (int(row["rank_U"]), len(observed)) != expected:
                            raise RuntimeError(
                                f"{s.name} failed I geometry in {row['x_region']}: "
                                f"rank={row['rank_U']}, |S|={len(observed)}, expected={expected}."
                            )
                        if not eval_codes.issubset(observed):
                            raise RuntimeError(f"{s.name}: I evaluation support is not locally observed.")
                        if not common_codes.issubset(observed):
                            raise RuntimeError(f"{s.name}: I common core is not locally observed.")

        except RuntimeError as exc:
            print(f"[seed {seed}] skipped: {exc}")
            continue

        regional_metadata = scenario_metadata.get("regional", {})
        regional_rank_map = (
            regional_metadata.get("rank_map")
            if isinstance(regional_metadata, Mapping)
            else None
        )
        catalog_rows.extend(scenario_catalog_rows(scenarios, regional_rank_map))
        print(
            f"[seed {seed}] support scenarios ready | count={len(scenarios)} | "
            f"population_eval={len(population_eval_idx)} | "
            f"preparation={time.perf_counter() - seed_started:.2f}s",
            flush=True,
        )

        scenario_list = list(scenarios.values())
        for model_position, (model_name, factory) in enumerate(
            factories.items(), start=1
        ):
            model_started = time.perf_counter()
            model_completed = 0
            model_seed = seed + 100_000 + MODEL_SEED_OFFSETS[model_name]
            print(
                f"\n[seed {seed}] model {model_position}/{len(factories)}: "
                f"{model_name} | scenarios={len(scenario_list)} | model_seed={model_seed}",
                flush=True,
            )
            for scenario_position, scenario in enumerate(scenario_list, start=1):
                if model_name == "T-Learner_RF":
                    observed = set(scenario.global_observed_support(data))
                    missing = sorted(set(range(N_REGIMES)).difference(observed))
                    if missing:
                        failure_rows.append({
                            "seed": seed,
                            "model": model_name,
                            "scenario": scenario.name,
                            "status": "unsupported",
                            "exception_type": "MissingTreatmentArms",
                            "message": (
                                "TLearnerMulti requires every treatment arm; "
                                f"missing categorical regimes {missing}."
                            ),
                        })
                        print(
                            f"  [{scenario_position:02d}/{len(scenario_list):02d}] "
                            f"{scenario.name}: UNSUPPORTED | missing_arms={missing}",
                            flush=True,
                        )
                        continue
                print(
                    f"  [{scenario_position:02d}/{len(scenario_list):02d}] "
                    f"{scenario.name}: training | n_train={scenario.n_training_rows()} | "
                    f"n_eval={len(scenario.test_idx)}",
                    flush=True,
                )
                try:
                    row, arows, crows, prows = evaluate_one_scenario(
                        data,
                        scenario,
                        population_eval_idx,
                        region_labels,
                        model_name,
                        factory,
                        model_seed,
                        training_outcome,
                    )
                except Exception as exc:
                    failure_rows.append({
                        "seed": seed,
                        "model": model_name,
                        "scenario": scenario.name,
                        "status": "failed",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    })
                    print(
                        f"  [{scenario_position:02d}/{len(scenario_list):02d}] "
                        f"{scenario.name}: FAILED | {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                row["rank_rotation"] = rotation
                metric_rows.append(row)
                ate_rows.extend(arows)
                cate_rows.extend(crows)
                policy_rows.extend(prows)
                model_completed += 1
                if scenario.family == "K_NEIGHBOR_SUPPORT_EXTRAPOLATION":
                    print(
                        f"  [{scenario_position:02d}/{len(scenario_list):02d}] "
                        f"{scenario.name}: done | fit={row['fit_seconds']:.2f}s | "
                        f"predict={row['target_predict_seconds']:.2f}s | "
                        f"K_weak_CATE={row['k_weak_all16_cate_rmse_origin']:.5f} | "
                        f"K_removed_CATE={row['k_weak_unsupported_cate_rmse_origin']:.5f} | "
                        f"K_regret={row['k_weak_all16_policy_regret']:.5f} | "
                        f"K_recovery={row['k_weak_all16_treatment_recovery_accuracy']:.3f} | "
                        f"neighbor_dS={row['k_neighbor_s_extra_mean']:.4f}",
                        flush=True,
                    )
                elif scenario.family == "K_UNSUPPORTED_MOD5_EXTRAPOLATION":
                    print(
                        f"  [{scenario_position:02d}/{len(scenario_list):02d}] "
                        f"{scenario.name}: done | fit={row['fit_seconds']:.2f}s | "
                        f"predict={row['target_predict_seconds']:.2f}s | "
                        f"KU_r4_CATE={row['ku_r4_all16_cate_rmse_origin']:.5f} | "
                        f"KU_r3_CATE={row['ku_r3_all16_cate_rmse_origin']:.5f} | "
                        f"KU_r2_CATE={row['ku_r2_all16_cate_rmse_origin']:.5f} | "
                        f"KU_r1_CATE={row['ku_r1_all16_cate_rmse_origin']:.5f} | "
                        f"KU_r1_recovery={row['ku_r1_all16_treatment_recovery_accuracy']:.3f}",
                        flush=True,
                    )
                elif scenario.family == "K_RANDOM_MOD8_RANK4_EXTRAPOLATION":
                    print(
                        f"  [{scenario_position:02d}/{len(scenario_list):02d}] "
                        f"{scenario.name}: done | fit={row['fit_seconds']:.2f}s | "
                        f"predict={row['target_predict_seconds']:.2f}s | "
                        f"KR_CATE={row['kr_overall_all16_cate_rmse_origin']:.5f} | "
                        f"KR_removed_CATE={row['kr_overall_unsupported_cate_rmse_origin']:.5f} | "
                        f"KR_regret={row['kr_overall_all16_policy_regret']:.5f} | "
                        f"KR_recovery={row['kr_overall_all16_treatment_recovery_accuracy']:.3f} | "
                        f"neighbor_dS={row['kr_neighbor_s_extra_mean']:.4f}",
                        flush=True,
                    )
                elif scenario.family == "K_RANDOM3_MOD8_RANK32_EXTRAPOLATION":
                    print(
                        f"  [{scenario_position:02d}/{len(scenario_list):02d}] "
                        f"{scenario.name}: done | fit={row['fit_seconds']:.2f}s | "
                        f"predict={row['target_predict_seconds']:.2f}s | "
                        f"KR3_CATE={row['kr3_overall_all16_cate_rmse_origin']:.5f} | "
                        f"KR3_removed_CATE={row['kr3_overall_unsupported_cate_rmse_origin']:.5f} | "
                        f"KR3_regret={row['kr3_overall_all16_policy_regret']:.5f} | "
                        f"KR3_recovery={row['kr3_overall_all16_treatment_recovery_accuracy']:.3f} | "
                        f"neighbor_dS={row['kr3_neighbor_s_extra_mean']:.4f}",
                        flush=True,
                    )
                elif scenario.family == "K_EXTREME_MAXMOD_EXTRAPOLATION":
                    variant = str(row["kext_variant"])
                    pp = "kext4" if variant == "RANK4_MAXMOD" else "kextlt4"
                    print(
                        f"  [{scenario_position:02d}/{len(scenario_list):02d}] "
                        f"{scenario.name}: done | fit={row['fit_seconds']:.2f}s | "
                        f"predict={row['target_predict_seconds']:.2f}s | "
                        f"KEXT_CATE={row[f'{pp}_overall_all16_cate_rmse_origin']:.5f} | "
                        f"KEXT_removed_CATE={row[f'{pp}_overall_unsupported_cate_rmse_origin']:.5f} | "
                        f"KEXT_regret={row[f'{pp}_overall_all16_policy_regret']:.5f} | "
                        f"KEXT_recovery={row[f'{pp}_overall_all16_treatment_recovery_accuracy']:.3f} | "
                        f"module={int(row['kext_effective_module'])}",
                        flush=True,
                    )
                elif scenario.family == "I_THEOREM_SAFE_SUPPORT_LADDER":
                    print(
                        f"  [{scenario_position:02d}/{len(scenario_list):02d}] "
                        f"{scenario.name}: done | fit={row['fit_seconds']:.2f}s | "
                        f"predict={row['target_predict_seconds'] + row['population_predict_seconds']:.2f}s | "
                        f"I_common_RMSE_mu={row['i_common_rmse_mu']:.5f} | "
                        f"I_common_CATE={row['i_common_cate_rmse_origin']:.5f} | "
                        f"I_common_ATE={row['i_common_mean_ate_abs_error']:.5f} | "
                        f"I_common_regret={row['i_common_policy_regret']:.5f} | "
                        f"I_common_recovery={row['i_common_treatment_recovery_accuracy']:.3f}",
                        flush=True,
                    )
                else:
                    print(
                        f"  [{scenario_position:02d}/{len(scenario_list):02d}] "
                        f"{scenario.name}: done | fit={row['fit_seconds']:.2f}s | "
                        f"predict={row['target_predict_seconds'] + row['population_predict_seconds']:.2f}s | "
                        f"RMSE_mu={row['target_rmse_mu_all']:.5f} | "
                        f"CATE_RMSE={row['population_cate_rmse_origin']:.5f} | "
                        f"ATE_MAE={row['population_mean_ate_abs_error']:.5f} | "
                        f"policy_regret={row['population_policy_regret']:.5f} | "
                        f"recovery={row['population_treatment_recovery_accuracy']:.3f} | "
                        f"hamming={row['population_treatment_recovery_hamming_distance']:.3f}",
                        flush=True,
                    )

            print(
                f"[seed {seed}] {model_name} finished | completed={model_completed}/"
                f"{len(scenario_list)} | elapsed={time.perf_counter() - model_started:.2f}s",
                flush=True,
            )

        successful += 1
        print(
            f"[seed {seed}] support ok ({successful}/{replicates}) | rotation={rotation} | "
            f"population_eval={len(population_eval_idx)} | scenarios={len(scenarios)} | "
            f"model_issues={sum(int(r['seed']) == seed for r in failure_rows)} | "
            f"elapsed={time.perf_counter() - seed_started:.2f}s"
        )

    if successful < replicates:
        raise RuntimeError(
            f"Only {successful}/{replicates} successful repetitions after {attempt} attempts."
        )

    failure_columns = [
        "seed", "model", "scenario", "status", "exception_type", "message"
    ]
    pd.DataFrame(failure_rows, columns=failure_columns).to_csv(
        outdir / "failures.csv", index=False
    )
    if not metric_rows:
        raise RuntimeError(
            "All model/scenario fits failed. See failures.csv for diagnostics."
        )

    metrics = pd.DataFrame(metric_rows)
    supports = pd.DataFrame(support_rows)
    metrics, relative_worsening = add_relative_worsening_vs_full2k(metrics)
    contrasts = build_paired_contrasts(
        metrics, n_boot=bootstrap_repetitions
    )
    summary = summarize_metrics(metrics)

    metrics.to_csv(outdir / "seed_metrics.csv", index=False)
    supports.to_csv(outdir / "local_support_diagnostics.csv", index=False)
    contrasts.to_csv(outdir / "paired_contrasts.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    relative_worsening.to_csv(outdir / "relative_worsening_vs_full2k.csv", index=False)
    pd.DataFrame(ate_rows).to_csv(outdir / "population_ate_by_treatment.csv", index=False)
    pd.DataFrame(cate_rows).to_csv(outdir / "population_cate_by_treatment.csv", index=False)
    policy_class_df = pd.DataFrame(policy_rows)
    policy_class_df.to_csv(outdir / "population_policy_class_metrics.csv", index=False)
    # Same class-level table under an explicit treatment-recovery name for article use.
    policy_class_df.to_csv(outdir / "population_treatment_recovery_by_class.csv", index=False)
    pd.DataFrame(catalog_rows).drop_duplicates().to_csv(outdir / "scenario_catalog.csv", index=False)
    i_metrics = metrics[metrics["family"] == "I_THEOREM_SAFE_SUPPORT_LADDER"].copy()
    i_metrics.to_csv(outdir / "i_theorem_safe_supported_only_metrics.csv", index=False)
    l_metrics = metrics[metrics["family"] == "L_RANK4_NORM_RICHNESS"].copy()
    l_metrics.to_csv(outdir / "l_rank4_norm_richness_metrics.csv", index=False)
    # J uses H FULL16 and H R4_CANONICAL as exact endpoints; save the full path explicitly.
    j_path_names = [
        "H_LOCAL_SUPPORT_LADDER_FULL16_DOM",
        "J_RANK4_SUPPORT12_DOM",
        "J_RANK4_SUPPORT8_DOM",
        "H_LOCAL_SUPPORT_LADDER_R4_CANONICAL_DOM",
    ]
    j_metrics = metrics[metrics["scenario"].isin(j_path_names)].copy()
    j_metrics.to_csv(outdir / "j_rank4_support_smoothing_metrics.csv", index=False)
    k_metrics = metrics[metrics["family"] == "K_NEIGHBOR_SUPPORT_EXTRAPOLATION"].copy()
    k_metrics.to_csv(outdir / "k_neighbor_extrapolation_metrics.csv", index=False)
    ku_metrics = metrics[metrics["family"] == "K_UNSUPPORTED_MOD5_EXTRAPOLATION"].copy()
    ku_metrics.to_csv(outdir / "k_unsupported_mod5_extrapolation_metrics.csv", index=False)
    kr_metrics = metrics[metrics["family"] == "K_RANDOM_MOD8_RANK4_EXTRAPOLATION"].copy()
    kr_metrics.to_csv(outdir / "k_random_mod8_rank4_extrapolation_metrics.csv", index=False)
    kr3_names = [K_RANDOM_SCENARIO_NAMES["FULL16_REFERENCE"], K_RANDOM3_SCENARIO_NAMES["MOD8"]]
    kr3_metrics = metrics[metrics["scenario"].isin(kr3_names)].copy()
    kr3_metrics.to_csv(outdir / "k_random3_mod8_rank32_extrapolation_metrics.csv", index=False)
    kext_names = [
        K_RANDOM_SCENARIO_NAMES["FULL16_REFERENCE"],
        K_EXTREME_SCENARIO_NAMES["RANK4"],
        K_EXTREME_SCENARIO_NAMES["LT4"],
    ]
    kext_metrics = metrics[metrics["scenario"].isin(kext_names)].copy()
    kext_metrics.to_csv(outdir / "k_extreme_maxmod_extrapolation_metrics.csv", index=False)

    plot_core(metrics, outdir / "core_support_validation.png")
    plot_relative_worsening(metrics, outdir / "relative_worsening_vs_full2k.png")
    plot_regional(metrics, "E_CANONICAL", outdir / "regional_canonical_cate.png")
    plot_regional(metrics, "F_GENERAL_BASIS", outdir / "regional_general_basis_cate.png")
    plot_regional(metrics, "G_GENERAL_REDUNDANT", outdir / "regional_general_redundant_cate.png")
    plot_l_rank4_norm_richness(metrics, outdir / "l_rank4_norm_richness_cate.png")
    plot_support_ladder(metrics, outdir / "h_local_support_ladder_cate.png")
    plot_j_rank4_support_smoothing(metrics, outdir / "j_rank4_support_smoothing_cate.png")
    plot_i_theorem_safe(metrics, outdir / "i_theorem_safe_common_core_cate.png")
    plot_k_neighbor_extrapolation(metrics, outdir / "k_neighbor_extrapolation_cate.png")
    plot_k_unsupported_mod5(metrics, outdir / "k_unsupported_mod5_cate.png")
    plot_k_random_mod8(metrics, outdir / "k_random_mod8_rank4_cate.png")
    plot_k_random3_mod8(metrics, outdir / "k_random3_mod8_rank32_cate.png")
    plot_k_extreme(metrics, outdir / "k_extreme_maxmod_cate.png")
    plot_treatment_recovery(metrics, outdir / "treatment_recovery_all_scenarios.png")

    unexpected_failures = [r for r in failure_rows if r["status"] == "failed"]
    if unexpected_failures:
        raise RuntimeError(
            f"{len(unexpected_failures)} unexpected model/scenario fit(s) failed. "
            f"Partial outputs were saved to {outdir.resolve()}; see failures.csv."
        )

    return metrics, supports, contrasts


def support_only_smoke_test(
    seed: int = BASE_SEED,
    max_train_n: int | None = MAX_TRAIN_N_DEFAULT,
    dominant_share: float = DOMINANT_SHARE_DEFAULT,
    sample_n: int = N,
    eval_fraction: float = POP_EVAL_FRACTION,
    selected_families: Sequence[str] = tuple(
        name for name in AVAILABLE_FAMILIES if name != "K"
    ),
) -> None:
    selected_families = expand_family_dependencies(selected_families)
    data = generate_base_dgp(seed, n=sample_n)
    region_labels, cuts = define_x_regions(data)
    train_pool, population_eval_idx = common_population_eval_split(
        region_labels, seed + 40_000, eval_fraction=eval_fraction
    )

    scenarios, scenario_metadata = build_selected_family_scenarios(
        data=data,
        train_pool=train_pool,
        population_eval_idx=population_eval_idx,
        region_labels=region_labels,
        rotation=0,
        seed=seed,
        dominant_share=dominant_share,
        max_train_n=max_train_n,
        selected_families=selected_families,
        cycle_index=0,
    )

    # Names retained for the detailed all-family diagnostic section below.
    abc_meta = scenario_metadata.get("ABC")
    d_meta = scenario_metadata.get("D")
    reg_meta = scenario_metadata.get("regional")
    l_meta = scenario_metadata.get("L")
    ladder_meta = scenario_metadata.get("H")
    j_meta = scenario_metadata.get("J")
    i_meta = scenario_metadata.get("I")
    k_meta = scenario_metadata.get("K_NEIGHBOR")
    ku_meta = scenario_metadata.get("K_MOD5")
    kr_meta = scenario_metadata.get("K_RANDOM")
    kr3_meta = scenario_metadata.get("K_RANDOM3")
    kext_meta = scenario_metadata.get("K_EXTREME")
    neighbor_extrapolation = {
        n: s for n, s in scenarios.items()
        if scenario_family_selector(s) == "K_NEIGHBOR"
    }
    k_extreme = {
        n: s for n, s in scenarios.items()
        if scenario_family_selector(s) == "K_EXTREME"
    }
    theorem_safe_ladder = {
        n: s for n, s in scenarios.items()
        if scenario_family_selector(s) == "I"
    }

    print(f"n={sample_n}, seed={seed}, eval_fraction={eval_fraction:.3f}")
    print("selected family closure:", ", ".join(selected_families))
    print("X cutpoints:", cuts)
    print("population eval n:", len(population_eval_idx))
    if isinstance(reg_meta, Mapping):
        print("rank map for regional smoke test:", reg_meta["rank_map"])
        print("regional E/F n:", reg_meta["common_train_n_EF"], "regional G n:", reg_meta["train_n_G"])
    if isinstance(abc_meta, Mapping):
        print("ABC common n:", abc_meta["common_train_n"])
    if isinstance(d_meta, Mapping):
        print("D common n:", d_meta["common_train_n"])
    if isinstance(l_meta, Mapping):
        print(
            "L rank4 norm-richness n:", l_meta["train_n"],
            "profile map:", l_meta["profile_map"],
            "dominant share:", l_meta["dominant_share"],
        )
    if isinstance(ladder_meta, Mapping):
        print(
            "H support ladder n:", ladder_meta["train_n"],
            "dominant region:", ladder_meta["dominant_region"],
            "allocation:", ladder_meta["allocation"],
            "omitted direction:", ladder_meta["omitted_direction_bits"],
            "redundant vertex:", ladder_meta["redundant_vertex_bits"],
        )
    print()

    for name, scenario in scenarios.items():
        rows = scenario_support_diagnostics_rows(data, scenario, region_labels)
        ranks = {r["x_region"]: r["rank_U"] for r in rows}
        eps0 = {r["x_region"]: round(r["baseline_share_hat"], 4) for r in rows}
        epsu = {r["x_region"]: round(r["epsilon_U_hat"], 4) for r in rows}
        exact = {r["x_region"]: r["all_2k_per_x"] for r in rows}
        print(
            f"{name:30s} n_rows={scenario.n_training_rows():6d} "
            f"n_unique_x={scenario.n_unique_training_x():5d} "
            f"ranks={ranks} all2K|x={exact} eps0={eps0} epsU={epsu}"
        )

    all_concrete_families = {name for name in AVAILABLE_FAMILIES if name != "K"}
    if set(selected_families) != all_concrete_families:
        return

    print("\nGeneral-basis supports:")
    for region in X_REGIONS:
        print(
            region,
            "rank", reg_meta["rank_map"][region],
            "support", bit_strings(reg_meta["F_supports"][region]),
            "basis", bit_strings(reg_meta["F_bases"][region]),
        )

    print("\nGeneral-redundant supports:")
    for region in X_REGIONS:
        print(
            region,
            "rank", reg_meta["rank_map"][region],
            "support", bit_strings(reg_meta["G_supports"][region]),
            "basis", bit_strings(reg_meta["G_bases"][region]),
        )

    print("\nL rank-4 size-5 norm-richness supports:")
    for region in X_REGIONS:
        profile = l_meta["profile_map"][region]
        support = tuple(l_meta["supports"][region])
        basis = tuple(l_meta["bases"][region])
        print(
            region,
            "profile", profile,
            "rank", rank_of_codes(support),
            "|S|", len(support),
            "weights", tuple(sorted(treatment_hamming_weight(c) for c in basis)),
            "support", bit_strings(support),
        )

    print("\nH local support ladder (dominant-region supports):")
    dom = ladder_meta["dominant_region"]
    for profile in SUPPORT_LADDER_ORDER:
        support = ladder_meta["profile_supports"][profile][dom]
        basis = ladder_meta["profile_bases"][profile][dom]
        print(
            profile,
            "region", dom,
            "rank", rank_of_codes(support),
            "|S|", len(support),
            "support", bit_strings(support),
            "basis", bit_strings(basis),
        )

    print("\nJ rank-4 support smoothing (H endpoints + two nested intermediate supports):")
    print(
        "dominant", j_meta["dominant_region"],
        "n", j_meta["train_n"],
        "path total support", [16, 12, 8, 5],
    )
    for label, support in [
        ("FULL16", j_meta["support16"]),
        ("SUPPORT12", j_meta["support12"]),
        ("SUPPORT8", j_meta["support8"]),
        ("4 BASIS + ORIGIN", j_meta["support5_canonical"]),
    ]:
        print(
            label,
            "rank", rank_of_codes(support),
            "|S|", len(support),
            "support", bit_strings(support),
        )

    print("\nK neighbor-X extrapolation (same anchors; every second X loses noncanonical rows):")
    print(
        "weak X", k_meta["k_n_weak_x"],
        "full-neighbor X", k_meta["k_n_full_x"],
        "reference rows", k_meta["reference_train_rows"],
        "mixed rows", k_meta["mixed_train_rows"],
        "removed fraction", round(k_meta["removed_fraction"], 4),
        "order", k_meta["k_order_direction"],
    )
    print(
        "neighbor gap mean |dS|", round(k_meta["k_neighbor_s_extra_mean"], 6),
        "mean Xstar L2", round(k_meta["k_neighbor_xstar_l2_mean"], 4),
    )
    for name in [
        K_NEIGHBOR_SCENARIO_NAMES["FULL16_REFERENCE"],
        K_NEIGHBOR_SCENARIO_NAMES["HALF_CANONICAL"],
    ]:
        scenario = neighbor_extrapolation[name]
        print(
            name,
            "profile", scenario.metadata["k_profile"],
            "weak support", bit_strings(
                tuple(range(N_REGIMES)) if scenario.metadata["k_profile"] == "FULL16_REFERENCE"
                else tuple(scenario.metadata["k_weak_support_codes"])
            ),
            "full-neighbor support", bit_strings(tuple(range(N_REGIMES))),
        )

    print("\nK-UNSUPPORTED modulo-5 extrapolation (interleaved full/r4/r3/r2/r1 X anchors):")
    print(
        "group sizes", ku_meta["ku_group_sizes"],
        "reference rows", ku_meta["reference_train_rows"],
        "mixed rows", ku_meta["mixed_train_rows"],
        "removed fraction", round(ku_meta["removed_fraction"], 4),
        "order", ku_meta["ku_order_direction"],
    )
    print(
        "nearest full mean |dS|", round(ku_meta["ku_neighbor_s_extra_mean"], 6),
        "nearest full mean Xstar L2", round(ku_meta["ku_neighbor_xstar_l2_mean"], 4),
    )
    for label in ("full", "r4", "r3", "r2", "r1"):
        support = tuple(ku_meta["ku_support_codes"][label])
        print(
            label,
            "mod", ku_meta["ku_mod_by_group"][label],
            "rank", rank_of_codes(support),
            "|S|", len(support),
            "support", bit_strings(support),
        )

    print("\nK-RANDOM modulo-8 diverse rank-4 supports:")
    print(
        "group sizes", kr_meta["kr_group_sizes"],
        "reference rows", kr_meta["reference_train_rows"],
        "mixed rows", kr_meta["mixed_train_rows"],
        "removed fraction", round(kr_meta["removed_fraction"], 4),
        "order", kr_meta["kr_order_direction"],
    )
    print(
        "union", bit_strings(kr_meta["kr_support_union_codes"]),
        "max nonzero overlap", kr_meta["kr_max_pairwise_nonzero_overlap"],
        "usage", kr_meta["kr_nonzero_usage"],
        "min internal Hamming diversity", kr_meta["kr_internal_hamming_diversity_min"],
    )
    print(
        "nearest complementary mean |dS|", round(kr_meta["kr_neighbor_s_extra_mean"], 6),
        "mean Xstar L2", round(kr_meta["kr_neighbor_xstar_l2_mean"], 4),
    )
    for mod in range(8):
        support = tuple(kr_meta["kr_support_codes"][mod])
        print(
            f"mod {mod}",
            "rank", rank_of_codes(support),
            "|S|", len(support),
            "support", bit_strings(support),
            "internal_hamming", _support_hamming_diversity(support),
        )


    print("\nK-RANDOM3 modulo-8 alternating rank-3/rank-2 supports:")
    print(
        "group sizes", kr3_meta["kr3_group_sizes"],
        "shared reference rows", kr3_meta["reference_train_rows"],
        "mixed rows", kr3_meta["mixed_train_rows"],
        "removed fraction", round(kr3_meta["removed_fraction"], 4),
        "order", kr3_meta["kr3_order_direction"],
    )
    print(
        "union", bit_strings(kr3_meta["kr3_support_union_codes"]),
        "max nonzero overlap", kr3_meta["kr3_max_pairwise_nonzero_overlap"],
        "usage", kr3_meta["kr3_nonzero_usage"],
        "total internal Hamming diversity", kr3_meta["kr3_total_hamming_diversity"],
    )
    for mod in range(8):
        support = tuple(kr3_meta["kr3_support_codes"][mod])
        print(
            f"mod {mod}",
            "rank", rank_of_codes(support),
            "|S|", len(support),
            "support", bit_strings(support),
            "internal_hamming", _support_hamming_diversity(support),
        )

    print("\nK-EXTREME maximum-module support variation:")
    print(
        "rank4 theoretical/effective module",
        kext_meta["rank4_theoretical_module"], kext_meta["rank4_effective_module"],
        "rank<4 theoretical/effective module",
        kext_meta["lt4_theoretical_module"], kext_meta["lt4_effective_module"],
    )
    for key in ("RANK4", "LT4"):
        scenario = k_extreme[K_EXTREME_SCENARIO_NAMES[key]]
        print(
            scenario.name,
            "rows", scenario.n_training_rows(),
            "unique supports", scenario.metadata["kext_n_unique_supports_used"],
            "rank counts", scenario.metadata["kext_rank_counts"],
            "cardinality counts", scenario.metadata["kext_cardinality_counts"],
        )

    print("\nI theorem-safe ladder (same support in all X regions; supported-only test):")
    print(
        "common core", bit_strings(i_meta["common_core"]),
        "omitted", i_meta["omitted_direction_bits"],
        "redundant", i_meta["redundant_vertex_bits"],
        "common n", i_meta["common_train_n"],
    )
    for profile in I_THEOREM_SAFE_ORDER:
        scenario = theorem_safe_ladder[I_THEOREM_SAFE_SCENARIO_NAMES[profile]]
        support = tuple(scenario.metadata["evaluation_support_codes"])
        print(
            profile,
            "rank", rank_of_codes(support),
            "|S|", len(support),
            "test support", bit_strings(support),
            "all X regions identical",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Controlled support validation using the causal estimators from "
            "models.py. Defaults come from config.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="JSON file; uses its 'base' object and ignores the suite grid.",
    )
    p.add_argument(
        "--replicates",
        type=int,
        default=None,
        help="Override config.json base.n_simulations.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config.json base.seed.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help=f"Override config.json base.n; must be at least {MIN_N}.",
    )
    p.add_argument(
        "--eval-fraction",
        type=float,
        default=None,
        help="Override config.json base.test_size for the common evaluation sample.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Output directory. By default, uses a dedicated subdirectory below "
            "config.json base.output_dir, resolved relative to this script."
        ),
    )
    model_group = p.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Run one model. Canonical names and aliases such as "
            "T_Learner_RF/T-Learner_RF are accepted."
        ),
    )
    model_group.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Run 'all' or a comma-separated list: "
            "S-Learner_RF,T-Learner_RF,HydraNet,MEDUSA."
        ),
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="Print selectable model names and exit.",
    )
    family_group = p.add_mutually_exclusive_group()
    family_group.add_argument(
        "--family",
        type=str,
        default=None,
        help=(
            "Run one test family, adding the minimal reference families needed "
            "for its comparisons; for example, L also runs H and BASE."
        ),
    )
    family_group.add_argument(
        "--families",
        type=str,
        default=None,
        help=(
            "Run 'all' or a comma-separated family list. Use --list-families "
            "to see the accepted names and automatic dependencies."
        ),
    )
    p.add_argument(
        "--list-families",
        action="store_true",
        help="Print selectable family names, descriptions, and dependencies, then exit.",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for HydraNet/MEDUSA; RF models always run on CPU.",
    )
    p.add_argument(
        "--verbose-models",
        action="store_true",
        help="Show periodic epoch/loss logs from HydraNet and MEDUSA.",
    )
    p.add_argument(
        "--rf-n-jobs",
        type=int,
        default=None,
        help=(
            "Random-Forest workers; -1 uses every available CPU and 1 disables "
            "parallel tree construction."
        ),
    )
    p.add_argument("--max-train-n", type=int, default=MAX_TRAIN_N_DEFAULT)
    p.add_argument("--dominant-region-share", type=float, default=DOMINANT_SHARE_DEFAULT)
    p.add_argument(
        "--training-outcome",
        choices=("contaminated", "clean"),
        default="clean",
        help=("clean is the primary theory-facing mode: it uses the pre-contamination factual "
              "Bernoulli label, whose conditional mean is mu(x,t). contaminated preserves the "
              "extra observation-noise stress from the source benchmark."),
    )
    p.add_argument("--support-only", action="store_true")
    p.add_argument(
        "--postprocess-only",
        action="store_true",
        help=(
            "Do not refit models. Load seed_metrics.csv from --outdir (or the "
            "configured default output directory), rebuild summary.csv, and "
            "regenerate all available figures."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_models:
        print("\n".join(AVAILABLE_MODELS))
        return
    if args.list_families:
        for family in AVAILABLE_FAMILIES:
            dependencies = (
                K_SUBFAMILIES
                if family == "K"
                else FAMILY_DEPENDENCIES.get(family, ())
            )
            suffix = (
                f" | includes: BASE, {', '.join(dependencies)}"
                if dependencies
                else " | includes: BASE"
                if family != "BASE"
                else ""
            )
            print(f"{family:12s} {FAMILY_DESCRIPTIONS[family]}{suffix}")
        print("ALL          all families")
        return

    input_defaults = load_input_defaults(args.config)
    effective_config = dict(input_defaults)
    if args.seed is not None:
        effective_config["seed"] = args.seed
    if args.n is not None:
        effective_config["n"] = args.n
    if args.eval_fraction is not None:
        effective_config["test_size"] = args.eval_fraction
    if args.verbose_models:
        effective_config["verbose_hydra"] = True
        effective_config["verbose_medusa"] = True
    if args.rf_n_jobs is not None:
        if args.rf_n_jobs == 0 or args.rf_n_jobs < -1:
            raise ValueError("--rf-n-jobs must be -1 or a positive integer.")
        effective_config["rf_n_jobs"] = args.rf_n_jobs
    configure_dgp_from_input(effective_config)

    replicates = (
        args.replicates
        if args.replicates is not None
        else int(effective_config.get("n_simulations", 10))
    )
    if replicates < 1:
        raise ValueError("--replicates must be at least 1.")
    if args.max_train_n is not None and args.max_train_n < 400:
        raise ValueError(
            "--max-train-n must be at least 400 for the regional scenarios."
        )
    if not 0.25 < args.dominant_region_share < 1.0:
        raise ValueError("--dominant-region-share must be in (0.25, 1.0).")

    model_value = (
        args.model
        if args.model is not None
        else args.models
        if args.models is not None
        else str(effective_config.get("models", "all"))
    )
    selected_models = normalize_model_selection(model_value)
    family_value = (
        args.family
        if args.family is not None
        else args.families
        if args.families is not None
        else "all"
    )
    requested_families = normalize_family_selection(family_value)
    selected_families = expand_family_dependencies(requested_families)
    has_neural_model = bool({"HydraNet", "MEDUSA"}.intersection(selected_models))
    if args.support_only:
        device = "not-used"
    elif has_neural_model:
        device = resolve_device(args.device)
    else:
        device = "cpu"

    if args.outdir is not None:
        outdir = args.outdir
    else:
        configured_output = Path(str(effective_config.get("output_dir", "final")))
        if not configured_output.is_absolute():
            configured_output = Path(__file__).resolve().parent / configured_output
        if requested_families == ("ALL",):
            results_subdir = DEFAULT_RESULTS_SUBDIR
        else:
            selection_slug = "_".join(name.lower() for name in requested_families)
            results_subdir = f"{DEFAULT_RESULTS_SUBDIR}_{selection_slug}"
        outdir = configured_output / results_subdir

    if args.postprocess_only:
        outdir.mkdir(parents=True, exist_ok=True)
        postprocess_existing_outputs(outdir)
        return

    print(f"Config: {Path(args.config).expanduser().resolve()}")
    print(f"Models: {', '.join(selected_models)}")
    print(f"Requested families: {', '.join(requested_families)}")
    print(f"Training family closure: {', '.join(selected_families)}")
    print(f"Neural device: {device} (RF estimators use CPU)")
    if {"S-Learner_RF", "T-Learner_RF"}.intersection(selected_models):
        rf_n_jobs = int(effective_config.get("rf_n_jobs", RF_N_JOBS_DEFAULT))
        print(
            "RF config: "
            f"trees={effective_config.get('rf_trees', 300)}, "
            f"min_samples_leaf={effective_config.get('rf_min_samples_leaf', 20)}, "
            f"n_jobs={rf_n_jobs}"
        )
    if "HydraNet" in selected_models:
        print(
            "HydraNet config: "
            f"epochs={effective_config.get('hydra_epochs', 200)}, "
            f"batch={effective_config.get('hydra_batch_size', 256)}, "
            f"lr={effective_config.get('hydra_lr', 1e-3)}"
        )
    if "MEDUSA" in selected_models:
        print(
            "MEDUSA config: "
            f"epochs={effective_config.get('medusa_epochs', 100)}, "
            f"batch={effective_config.get('medusa_batch_size', 256)}, "
            f"lr={effective_config.get('medusa_lr', 8e-4)}"
        )
    if "T-Learner_RF" in selected_models:
        print(
            "T-Learner note: scenarios missing any of the 16 arms are marked "
            "unsupported; no counterfactual heads are fabricated."
        )
    print(
        f"Defaults: replicates={replicates}, seed={BASE_SEED}, n={N}, "
        f"eval_fraction={POP_EVAL_FRACTION}"
    )

    if args.support_only:
        support_only_smoke_test(
            seed=BASE_SEED,
            max_train_n=args.max_train_n,
            dominant_share=args.dominant_region_share,
            sample_n=N,
            eval_fraction=POP_EVAL_FRACTION,
            selected_families=selected_families,
        )
        return

    outdir.mkdir(parents=True, exist_ok=True)
    resolved = {
        "input_file": str(Path(args.config).expanduser().resolve()),
        "input_base": effective_config,
        "controlled_validation": {
            "replicates": replicates,
            "seed": BASE_SEED,
            "n": N,
            "eval_fraction": POP_EVAL_FRACTION,
            "models": list(selected_models),
            "rf_n_jobs": int(effective_config.get("rf_n_jobs", RF_N_JOBS_DEFAULT)),
            "requested_families": list(requested_families),
            "training_family_closure": list(selected_families),
            "requested_device": args.device,
            "resolved_neural_device": device,
            "verbose_models": args.verbose_models,
            "max_train_n": args.max_train_n,
            "dominant_region_share": args.dominant_region_share,
            "training_outcome": args.training_outcome,
            "output_dir": str(outdir),
        },
    }
    (outdir / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    experiment_started = time.perf_counter()
    metrics, supports, contrasts = run_experiment(
        replicates=replicates,
        outdir=outdir,
        max_train_n=args.max_train_n,
        dominant_share=args.dominant_region_share,
        training_outcome=args.training_outcome,
        model_config=effective_config,
        selected_models=selected_models,
        selected_families=selected_families,
        device=device,
        base_seed=BASE_SEED,
        sample_n=N,
        eval_fraction=POP_EVAL_FRACTION,
    )
    terminal_summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            completed_fits=("scenario", "count"),
            mean_rmse_mu=("target_rmse_mu_all", "mean"),
            mean_cate_rmse=("population_cate_rmse_origin", "mean"),
            mean_ate_mae=("population_mean_ate_abs_error", "mean"),
            mean_policy_regret=("population_policy_regret", "mean"),
            mean_policy_accuracy=("population_policy_accuracy", "mean"),
            mean_treatment_recovery=("population_treatment_recovery_accuracy", "mean"),
            mean_treatment_hamming=("population_treatment_recovery_hamming_distance", "mean"),
            mean_treatment_bit_accuracy=("population_treatment_recovery_bit_accuracy", "mean"),
            total_fit_seconds=("fit_seconds", "sum"),
        )
    )
    print("\n=== Model performance summary ===")
    print(terminal_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== Paired contrasts ===")
    print(contrasts.to_string(index=False))
    print(f"\nTotal elapsed: {time.perf_counter() - experiment_started:.2f}s")
    print(f"\nSaved to {outdir.resolve()}")


if __name__ == "__main__":
    main()
