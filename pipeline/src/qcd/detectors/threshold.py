"""ξ (xi) threshold handling for CDD (paper §4.4's "Threshold handling (ξ)"):

- `constants.CDD_XI_FIXED = 0.01` is the original CDD paper's 7B-calibrated,
  pre-fixed threshold — `classify()` uses this by default.
- **AUC (Q1b's primary metric) is threshold-independent, so ξ recalibration
  is not required for Q1b at all.**
- If a fixed threshold's point-accuracy is reported as a secondary
  descriptive statistic, arXiv:2603.03203 warns that re-selecting ξ by
  maximizing Youden's J **on the same evaluation set being scored** is an
  optimistic oracle — the paper's own words: "gives CDD every advantage."
  `select_threshold_youden` below is for a held-out calibration split only,
  and raises if the caller passes overlapping item ids between calibration
  and evaluation, rather than relying on the caller to remember not to.
"""

from __future__ import annotations

from qcd.constants import CDD_XI_FIXED


class ThresholdSelectionOnEvalSetError(Exception):
    """Raised when select_threshold_youden's calibration set overlaps the
    evaluation set — exactly the oracle-bias failure mode arXiv:2603.03203
    flags in its own re-selected-ξ methodology."""


def classify(scores: list[float], *, xi: float = CDD_XI_FIXED) -> list[bool]:
    """Applies a pre-fixed threshold. Default is the original CDD paper's
    ξ=0.01, calibrated on 7B models — do not re-select this per condition on
    the evaluation set (see select_threshold_youden's guard)."""
    return [score > xi for score in scores]


def youden_j(scores: list[float], labels: list[bool], threshold: float) -> float:
    """Youden's J = sensitivity + specificity - 1 at a given threshold."""
    tp = fn = tn = fp = 0
    for score, label in zip(scores, labels):
        predicted = score > threshold
        if label:
            tp += predicted
            fn += not predicted
        else:
            fp += predicted
            tn += not predicted
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return sensitivity + specificity - 1


def select_threshold_youden(
    calibration_scores: list[float],
    calibration_labels: list[bool],
    *,
    calibration_item_ids: list[str] | None = None,
    evaluation_item_ids: list[str] | None = None,
) -> tuple[float, float]:
    """Selects ξ by maximizing Youden's J over `calibration_scores` — this
    MUST be a held-out split, never the set point-accuracy is later reported
    against. If both id lists are supplied and overlap, raises
    `ThresholdSelectionOnEvalSetError` rather than silently reproducing
    arXiv:2603.03203's own flagged optimistic-oracle bias.

    Returns (best_threshold, best_j).
    """
    if calibration_item_ids is not None and evaluation_item_ids is not None:
        overlap = set(calibration_item_ids) & set(evaluation_item_ids)
        if overlap:
            raise ThresholdSelectionOnEvalSetError(
                f"{len(overlap)} item id(s) appear in both the calibration and "
                "evaluation sets — selecting xi by Youden's J on (part of) the "
                "same set being evaluated reproduces arXiv:2603.03203's own "
                "flagged optimistic-oracle bias ('gives CDD every advantage'). "
                "Use a disjoint held-out split, or classify() with a pre-fixed xi."
            )

    if not calibration_scores:
        raise ValueError("select_threshold_youden needs at least one calibration score")

    candidates = sorted(set(calibration_scores))
    best_threshold, best_j = candidates[0], -1.0
    for threshold in candidates:
        j = youden_j(calibration_scores, calibration_labels, threshold)
        if j > best_j:
            best_threshold, best_j = threshold, j
    return best_threshold, best_j
