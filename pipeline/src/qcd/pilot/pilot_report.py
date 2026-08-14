"""The five quantities paper §4.7 requires *together* before finalizing
sample sizes — "measuring only one leaves the plan unable to locate itself
within the tables in §4.5":

  (a) Q1a detector-score shift size d
  (b) Q1b's observed AUC and the cross-precision AUC correlation r
  (c) Q2's log-odds effect size and item-level correlation r
  (d) each condition's actual base-rate accuracy, per model
  (e) the proxy-label error rate e on the Olmo3 arm

This module exposes the statistical primitive for each quantity
(`cohens_d_paired`, `pearson_r`, `base_rate`, `proxy_label_error_rate`) plus
a `PilotReport` container — the actual gluing-together of raw per-item data
into a `PilotReport` happens in the caller (scripts/run_dry_run.py today, a
real pilot driver later), since this module shouldn't need to know the raw
data's storage format.
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
    """Actual base-rate accuracy for a (model, condition) cell — paper §4.7
    item (d), "measured per model in the pilot", never assumed to match
    another model's illustrative value (§4.5.3)."""
    outcomes = np.asarray(outcomes, dtype=float)
    if len(outcomes) == 0:
        raise ValueError("base_rate needs at least one outcome")
    return float(outcomes.mean())


def proxy_label_error_rate(proxy_labels, ground_truth_labels) -> float:
    """The measured *e* for the Olmo3 arm (§4.5.2, §5 step 5): fraction of
    items where the pre/post-cutoff proxy label disagrees with the
    corpus-search ground truth. Every other arm's *e* stays an assumption
    (§4.5.2's sensitivity table); this is the one arm where it's measured."""
    proxy = np.asarray(proxy_labels, dtype=bool)
    truth = np.asarray(ground_truth_labels, dtype=bool)
    if len(proxy) != len(truth):
        raise ValueError(f"proxy_label_error_rate needs equal-length arrays, got {len(proxy)} and {len(truth)}")
    if len(proxy) == 0:
        raise ValueError("proxy_label_error_rate needs at least one item")
    return float(np.mean(proxy != truth))


@dataclasses.dataclass
class PilotReport:
    # (a) per-detector Cohen's d of the paired fp16->target-precision score shift
    q1a_effect_size_d: dict[str, float] = dataclasses.field(default_factory=dict)
    # (b) per-detector baseline (fp16) AUC and cross-precision AUC correlation r
    q1b_baseline_auc: dict[str, float] = dataclasses.field(default_factory=dict)
    q1b_cross_precision_r: dict[str, float] = dataclasses.field(default_factory=dict)
    # (c) Q2's log-odds interaction effect and item-level correlation r
    q2_log_odds_effect: float | None = None
    q2_item_level_r: float | None = None
    # (d) per (model, dataset) base-rate accuracy
    base_rates: dict[tuple[str, str], float] = dataclasses.field(default_factory=dict)
    # (e) Olmo3 arm's measured proxy-label error rate
    olmo3_proxy_label_error_rate: float | None = None
