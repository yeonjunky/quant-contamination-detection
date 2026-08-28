"""Paired AUC + Hanley-McNeil SE, and the CDD pilot-gate ΔAUC model. Every
formula here was independently re-derived and checked against
paper/paper_draft.md's worked tables (§4.5.2, §4.6) during pipeline
construction, not copied from memory of "the standard formula" — see the
docstrings below for the exact table cells each function reproduces.

Empirical AUC itself (given real detector scores + labels) is the standard
rank-based estimator; only the analytic SE/power machinery is re-derived
here, since that's what the paper's tables depend on.
"""

from __future__ import annotations

import numpy as np

from qcd.analysis._stats import bisect, normal_cdf, normal_ppf, z_for_two_sided_test
from qcd.constants import ALPHA, POWER_TARGET


def empirical_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U / rank-sum AUC estimator: P(score of a random positive
    > score of a random negative), ties counted as one-half. `labels` is
    boolean (True = contaminated)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("empirical_auc needs at least one positive and one negative item")
    # Broadcast comparison: for each (pos, neg) pair, 1 if pos>neg, 0.5 if tie.
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def hanley_mcneil_se(auc: float, n: int) -> float:
    """SE(AUC) under equal-sized positive/negative groups (n1=n2=n), the
    Hanley & McNeil (1982) formula. Reproduces paper §4.5.2's SE(AUC) column
    exactly at AUC=0.70: n=164->0.029, 300->0.021, 542->0.016, 1000->0.012
    (verified during pipeline construction; see tests/test_auc_hanley_mcneil.py).
    """
    return hanley_mcneil_se_unequal(auc, n, n)


def hanley_mcneil_se_unequal(auc: float, n_positive: int, n_negative: int) -> float:
    """Hanley–McNeil SE for unequal positive and negative group sizes."""
    if n_positive < 2 or n_negative < 2:
        raise ValueError("AUC SE needs at least two items in each label group")
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (
        auc * (1 - auc)
        + (n_positive - 1) * (q1 - auc**2)
        + (n_negative - 1) * (q2 - auc**2)
    ) / (n_positive * n_negative)
    return float(np.sqrt(var))


def paired_auc_detection_limit(n: int, auc: float, r: float, alpha: float = ALPHA, power: float = POWER_TARGET) -> float:
    """Minimum detectable |ΔAUC| between two paired (same-item) AUC
    measurements with cross-precision correlation r, at n items per group.
    Reproduces §4.5.2's detection-limit columns exactly at AUC=0.70:
    r=0 -> 0.114/0.084/0.063/0.046, r=0.8 -> 0.051/0.038/0.028/0.021,
    r=0.9 -> 0.036/0.027/0.020/0.015 for n=164/300/542/1000.
    """
    return paired_auc_detection_limit_unequal(n, n, auc, r, alpha, power)


def paired_auc_detection_limit_unequal(
    n_positive: int,
    n_negative: int,
    auc: float,
    r: float,
    alpha: float = ALPHA,
    power: float = POWER_TARGET,
) -> float:
    """Paired-AUC detection limit for the actual two label-group sizes."""
    if not -1 <= r <= 1:
        raise ValueError("paired-AUC correlation r must be in [-1, 1]")
    se = hanley_mcneil_se_unequal(auc, n_positive, n_negative)
    se_diff = np.sqrt(2 * se**2 * (1 - r))
    return float(z_for_two_sided_test(alpha, power) * se_diff)


def items_needed_for_delta_auc(target_delta_auc: float, auc: float, r: float, alpha: float = ALPHA, power: float = POWER_TARGET) -> int:
    """Inverse of paired_auc_detection_limit: smallest n whose detection
    limit is <= target_delta_auc. Reproduces the label-noise table (§4.5.2)
    at auc=0.70, r=0.8: target=0.050->170, 0.040->287, 0.030->541, 0.020->1268."""

    def f(n: float) -> float:
        return paired_auc_detection_limit(max(int(round(n)), 2), auc, r, alpha, power) - target_delta_auc

    # f is decreasing in n; bisect() expects an increasing-or-decreasing
    # bracket and infers direction from the endpoints itself.
    n = bisect(f, 2, 1_000_000, iters=60)
    return int(np.ceil(n))


def quantization_delta_auc(auc: float, reduction: float) -> float:
    """§4.6's "assumed `reduction` reduction in separation" model, applied on
    the binormal d'-scale (d' = sqrt(2) * Phi^-1(AUC)) rather than to
    (AUC-0.5) directly. Reproduces the gate table's ΔAUC column exactly:
    AUC=0.52->0.002, 0.60->0.010, 0.70->0.019, 0.85->0.026, 0.95->0.019 at
    reduction=0.10 (verified during pipeline construction against a
    (AUC-0.5)*reduction hypothesis, which does *not* reproduce the table).

    `auc` is clamped away from {0, 1} before the Phi^-1 call: a *measured*
    AUC on a small pilot sample can legitimately come back as exactly 1.0
    (perfect separation), where Phi^-1 is undefined — found when the mock
    dry run's small item count produced exactly this (pipeline
    construction). The clamp keeps the gate check from crashing on real
    small-n pilot data instead of only ever seeing this in a mock."""
    auc = min(max(auc, 1e-9), 1 - 1e-9)
    d_prime = np.sqrt(2) * normal_ppf(auc)
    d_prime_reduced = (1 - reduction) * d_prime
    auc_reduced = normal_cdf(d_prime_reduced / np.sqrt(2))
    return float(auc - auc_reduced)


def cdd_gate_breakeven_auc(
    n: int = 542,
    r: float = 0.8,
    reduction: float = 0.10,
    alpha: float = ALPHA,
    power: float = POWER_TARGET,
) -> float:
    """Solves for the baseline AUC where quantization_delta_auc equals
    paired_auc_detection_limit — the pilot gate's break-even point.
    Reproduces CLAUDE.md §4.1 / paper §4.6's "≈0.79" (exact value 0.7936)."""

    def f(auc: float) -> float:
        return quantization_delta_auc(auc, reduction) - paired_auc_detection_limit(n, auc, r, alpha, power)

    return bisect(f, 0.5001, 0.9999, iters=60)
