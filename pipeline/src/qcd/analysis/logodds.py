"""The base-rate-confound model, paper §4.5.3 — CLAUDE.md §5 point 3's named
invariant ("Q2는 로그오즈 스케일로"). Item-conditional accuracy is modeled as
`sigmoid(alpha + sigma*Z - beta)`, `Z ~ N(0,1)` a per-item difficulty random
effect (`constants.DIFFICULTY_SIGMA`), `alpha` a condition intercept solved
to hit a target marginal base rate, `beta` the item-conditional log-odds
quantization drop. Marginalizing over `Z` (not just plugging in the mean)
is what reproduces the paper's own worked spurious-interaction table.

All three functions below were independently re-derived and checked against
paper/paper_draft.md's §4.5.3 table during this pipeline's construction (see
review/review_findings_round8.md's `verify_stats.py` for the standalone
version this module supersedes) — reproduces β=0.50 → 5.6pp/7.7pp/−2.2pp
exactly (tests/test_logodds.py).
"""

from __future__ import annotations

import numpy as np

from qcd.analysis._stats import bisect
from qcd.constants import DIFFICULTY_SIGMA

_Z_RANGE = 8.0
_Z_POINTS = 200_001


def _z_grid(z_range: float = _Z_RANGE, n_points: int = _Z_POINTS) -> tuple[np.ndarray, np.ndarray]:
    z = np.linspace(-z_range, z_range, n_points)
    weights = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    # Manual trapezoidal normalization (not np.trapz/np.trapezoid, whose
    # names moved across numpy 1.x/2.x — avoid the version dependency).
    dx = z[1] - z[0]
    total = dx * (np.sum(weights) - 0.5 * weights[0] - 0.5 * weights[-1])
    return z, weights / total


def _trapz(y: np.ndarray, dx: float) -> float:
    return float(dx * (np.sum(y) - 0.5 * y[0] - 0.5 * y[-1]))


def marginal_accuracy(alpha: float, sigma: float = DIFFICULTY_SIGMA, beta: float = 0.0) -> float:
    """E_Z[sigmoid(alpha + sigma*Z - beta)] — the marginal (population-
    average) accuracy for a condition with intercept `alpha`, difficulty SD
    `sigma`, and item-conditional log-odds drop `beta` (beta=0 -> baseline
    accuracy at this intercept, before any quantization effect)."""
    z, weights = _z_grid()
    p = 1 / (1 + np.exp(-(alpha + sigma * z - beta)))
    dx = z[1] - z[0]
    return _trapz(p * weights, dx)


def solve_intercept_for_base_rate(target_marginal_accuracy: float, sigma: float = DIFFICULTY_SIGMA) -> float:
    """Bisection for the condition intercept `alpha` whose marginal
    (beta=0) accuracy equals `target_marginal_accuracy` — e.g. HumanEval's
    illustrative 0.85 or LCB-post's 0.35 (constants.py)."""
    if not 0.0 < target_marginal_accuracy < 1.0:
        raise ValueError(f"target_marginal_accuracy must be in (0, 1), got {target_marginal_accuracy}")

    def f(alpha: float) -> float:
        return marginal_accuracy(alpha, sigma) - target_marginal_accuracy

    return bisect(f, -10.0, 10.0, iters=80)


def spurious_pp_interaction(
    beta: float,
    *,
    humaneval_base_rate: float,
    lcb_post_base_rate: float,
    sigma: float = DIFFICULTY_SIGMA,
) -> dict[str, float]:
    """The §4.5.3 worked-table row for a given item-conditional log-odds
    drop `beta`: percentage-point drop in each condition, and the resulting
    *spurious* interaction that appears purely from the two conditions'
    different base rates even though `beta` (the true, item-conditional
    effect) is identical across both. Returns pp values (not fractions).
    """
    alpha_he = solve_intercept_for_base_rate(humaneval_base_rate, sigma)
    alpha_lcb = solve_intercept_for_base_rate(lcb_post_base_rate, sigma)

    he_before, he_after = marginal_accuracy(alpha_he, sigma), marginal_accuracy(alpha_he, sigma, beta)
    lcb_before, lcb_after = marginal_accuracy(alpha_lcb, sigma), marginal_accuracy(alpha_lcb, sigma, beta)

    he_drop_pp = (he_before - he_after) * 100
    lcb_drop_pp = (lcb_before - lcb_after) * 100
    return {
        "humaneval_drop_pp": he_drop_pp,
        "lcb_post_drop_pp": lcb_drop_pp,
        "spurious_interaction_pp": he_drop_pp - lcb_drop_pp,
    }


def implied_correlation_at_p50(sigma: float = DIFFICULTY_SIGMA) -> float:
    """Cross-precision item-level correlation implied by the difficulty
    random-effect model at p=0.5, beta=0 — reproduces
    `constants.IMPLIED_R_AT_P50` (0.293089) via `r = (E[p^2] - 0.25) / 0.25`
    at the intercept solving marginal_accuracy(alpha, sigma) == 0.5 (alpha=0
    by symmetry)."""
    z, weights = _z_grid()
    p = 1 / (1 + np.exp(-(sigma * z)))
    dx = z[1] - z[0]
    e_p2 = _trapz(p**2 * weights, dx)
    return (e_p2 - 0.25) / 0.25


def aggregate_log_odds_bias(
    beta: float,
    *,
    humaneval_base_rate: float,
    lcb_post_base_rate: float,
    sigma: float = DIFFICULTY_SIGMA,
) -> float:
    """§4.5.3's residual bias from computing the interaction term on
    *aggregated* (marginal) log-odds rather than fitting the full item-
    conditional mixed model — the quantity the paper reports as "up to
    about 0.023" (max over beta in the worked table's range)."""
    alpha_he = solve_intercept_for_base_rate(humaneval_base_rate, sigma)
    alpha_lcb = solve_intercept_for_base_rate(lcb_post_base_rate, sigma)

    def logit(p: float) -> float:
        return float(np.log(p / (1 - p)))

    he_before, he_after = marginal_accuracy(alpha_he, sigma), marginal_accuracy(alpha_he, sigma, beta)
    lcb_before, lcb_after = marginal_accuracy(alpha_lcb, sigma), marginal_accuracy(alpha_lcb, sigma, beta)
    return (logit(he_before) - logit(he_after)) - (logit(lcb_before) - logit(lcb_after))
