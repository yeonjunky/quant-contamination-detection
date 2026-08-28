"""Turns analysis/auc.py's and analysis/_stats.py's primitives into the
specific power/sample-size tables paper §4.5 reports, parametrized by
explicit effect sizes/correlations rather than silently substituting observed
validation values. Every function reproduces a specific worked
table in the paper when called with the paper's own illustrative values
(tests/test_power.py).
"""

from __future__ import annotations

import numpy as np

from qcd.analysis._stats import bisect, normal_cdf, normal_ppf, z_for_two_sided_test
from qcd.analysis.auc import items_needed_for_delta_auc
from qcd.constants import ALPHA, POWER_TARGET


def items_needed_paired_t(d: float, alpha: float = ALPHA, power: float = POWER_TARGET) -> int:
    """Q1a (§4.5.1): items needed for a paired comparison at Cohen's d,
    n = ((z_{alpha/2}+z_power)/d)^2, rounded to nearest (matches the paper's
    own displayed 87/196 for d=0.3/0.2, computed exactly as 87.2/196.2)."""
    z = z_for_two_sided_test(alpha, power)
    return round((z / d) ** 2)


def power_diff_in_diff(n_per_condition: int, delta_pp: float, *, p: float = 0.5, alpha: float = ALPHA) -> float:
    """Q2 (§4.5.3): power of a 4-cell unpaired difference-in-differences at
    `n_per_condition` items, detecting a `delta_pp` percentage-point effect,
    under the paper's own stated formula:

        power = Phi(delta/SE - z_{alpha/2}) + Phi(-delta/SE - z_{alpha/2})
        SE = sqrt(4 * p * (1-p) / n)
    """
    delta = delta_pp / 100
    se = np.sqrt(4 * p * (1 - p) / n_per_condition)
    z_a2 = normal_ppf(1 - alpha / 2)
    return float(normal_cdf(delta / se - z_a2) + normal_cdf(-delta / se - z_a2))


def items_needed_diff_in_diff(
    delta_pp: float, *, p: float = 0.5, alpha: float = ALPHA, power_target: float = POWER_TARGET
) -> int:
    """Inverse of power_diff_in_diff: smallest n_per_condition reaching
    `power_target`. Reproduces §4.5.3's 196/785/3,140 at delta_pp=20/10/5."""

    def f(n: float) -> float:
        return power_diff_in_diff(max(int(round(n)), 2), delta_pp, p=p, alpha=alpha) - power_target

    n = bisect(f, 2, 10_000_000, iters=60)
    return int(np.ceil(n))


def minimum_detectable_diff_in_diff_unequal(
    n_suspect: int | float,
    n_control: int | float,
    *,
    p: float = 0.5,
    alpha: float = ALPHA,
    power: float = POWER_TARGET,
) -> float:
    """Q2's unpaired MDE in percentage points for unequal condition sizes.

    Each condition contributes one full-precision and one quantized cell, so
    ``SE² = 2p(1-p)/n_suspect + 2p(1-p)/n_control``. ``np.inf`` is accepted
    for the best-case bound where one condition is unlimited.
    """
    if n_suspect <= 0 or n_control <= 0:
        raise ValueError("Q2 condition sizes must be positive")
    variance = 2 * p * (1 - p) * (1 / n_suspect + 1 / n_control)
    return float(z_for_two_sided_test(alpha, power) * np.sqrt(variance) * 100)


def items_needed_with_label_noise(
    true_delta_auc: float,
    proxy_error_rate_e: float,
    *,
    true_auc: float,
    r: float,
    alpha: float = ALPHA,
    power: float = POWER_TARGET,
) -> int:
    """§4.5.2's label-noise table — CLAUDE.md §4.1's adopted "SE도 감쇠"
    column, not the non-adopted "SE 고정" column. Label noise attenuates
    *both* the target AUC difference **and** the observed AUC baseline
    itself by `(1 - 2e)` (`AUC_obs - 0.5 = (1-2e)(AUC_true - 0.5)`) — the SE
    used for sizing must be evaluated at the attenuated *observed* AUC, not
    the true one. Passing only the attenuated delta while leaving `auc`
    unattenuated (an earlier version of this function did exactly that)
    silently reproduces the non-adopted "SE 고정" column instead (found and
    fixed during pipeline construction: 266/472/1060 instead of the
    adopted 287/541/1268 — see tests/test_power.py).

    Reproduces 170/287/541/1268 (±1, a pre-existing rounding-convention
    ambiguity already documented in review/review_findings_round7.md) at
    e=0/0.10/0.20/0.30 for true_delta_auc=0.050, true_auc=0.70, r=0.8."""
    attenuation = 1 - 2 * proxy_error_rate_e
    observed_delta = attenuation * true_delta_auc
    observed_auc = 0.5 + attenuation * (true_auc - 0.5)
    return items_needed_for_delta_auc(observed_delta, observed_auc, r, alpha, power)
