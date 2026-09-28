"""Synthetic validation diagnostics used by the GPU-free dry run.

These primitives exercise the intended statistical data shapes, but values
computed from mock or smoke-test rows are engineering observations only. They
must not be used to resize the study, change confirmatory eligibility, or be
reported as manuscript results.

This module exposes the statistical primitive for each quantity
(`cohens_d_paired`, `pearson_r`, `base_rate`) plus a
`ValidationDiagnostics` container
— the actual gluing-together of raw per-item data
into diagnostics happens in the caller. The module does not assign scientific
status to those values and does not know the raw data's storage format.
"""

from __future__ import annotations

import dataclasses

import numpy as np


def cohens_d_paired(before, after) -> float:
    """Paired Cohen's d = mean(after - before) / SD(after - before) — Q1a's
    effect size (§4.5.1), computed from the same item's detector score at
    two precisions."""
    diff = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    if len(diff) < 2:
        raise ValueError("cohens_d_paired needs at least 2 paired observations")
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)


def pearson_r(x, y) -> float:
    """Cross-precision correlation (Q1b's r, Q2's item-level r) — plain
    Pearson correlation, numpy-only."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"pearson_r needs equal-length arrays, got {len(x)} and {len(y)}")
    if len(x) < 2:
        raise ValueError("pearson_r needs at least 2 paired observations")
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def base_rate(outcomes) -> float:
    """Development-only base-rate diagnostic for a model-condition cell.

    It is never assumed to match another model's illustrative value (§4.5.3).
    """
    outcomes = np.asarray(outcomes, dtype=float)
    if len(outcomes) == 0:
        raise ValueError("base_rate needs at least one outcome")
    return float(outcomes.mean())


@dataclasses.dataclass
class ValidationDiagnostics:
    # (a) per-detector Cohen's d of the paired bf16->target-precision score shift
    q1a_effect_size_d: dict[str, float] = dataclasses.field(default_factory=dict)
    # (b) per-detector baseline (bf16) AUC and cross-precision AUC correlation r
    q1b_baseline_auc: dict[str, float] = dataclasses.field(default_factory=dict)
    q1b_cross_precision_r: dict[str, float] = dataclasses.field(default_factory=dict)
    # (c) Q2's log-odds interaction effect and item-level correlation r
    q2_log_odds_effect: float | None = None
    q2_item_level_r: float | None = None
    # (d) per (model, dataset) base-rate accuracy
    base_rates: dict[tuple[str, str], float] = dataclasses.field(default_factory=dict)
