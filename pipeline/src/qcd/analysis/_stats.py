"""Shared numpy-only statistical primitives. No scipy (CLAUDE.md §6: "scipy
없음 — numpy + 직접 구현(이분법 등)으로"). The standard normal CDF is built on
`math.erf` (Python stdlib, exact); its inverse is obtained by bisection on
that CDF, exactly the "Hanley–McNeil SE 공식과 이분 탐색" recipe CLAUDE.md
names as sufficient to reproduce every table in §4.1.
"""

from __future__ import annotations

import math


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def normal_ppf(p: float, lo: float = -8.0, hi: float = 8.0, iters: int = 100) -> float:
    """Inverse standard normal CDF via bisection. `iters=100` on an initial
    bracket of width 16 gives resolution far below any precision these
    tables report at."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    for _ in range(iters):
        mid = (lo + hi) / 2
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def bisect(f, lo: float, hi: float, iters: int = 100) -> float:
    """Generic bisection for a monotonically increasing f with f(lo) < 0 <
    f(hi) (or the reverse — direction is inferred from the sign at `lo`)."""
    f_lo = f(lo)
    increasing = f_lo < f(hi)
    for _ in range(iters):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        below = f_mid < 0
        if below == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def z_for_two_sided_test(alpha: float, power: float) -> float:
    """z_{1-alpha/2} + z_{power}, the multiplier that turns a standard error
    into a minimum-detectable-effect at the given alpha/power (matches
    CLAUDE.md's named constant 2.8016 at alpha=0.05, power=0.80)."""
    return normal_ppf(1 - alpha / 2) + normal_ppf(power)
