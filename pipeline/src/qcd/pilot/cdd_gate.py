"""§4.6's pilot go/no-go gate for CDD: measure CDD's 16-bit baseline AUC in
the pilot; if it's below `constants.CDD_GATE_AUC` (0.7936, "≈0.79" in the
paper's prose), drop CDD from the primary analysis and report Q1 using the
probability-based detectors only, with CDD's (in)ability to function at
this scale reported as a standalone finding (Contribution 4, §1).

The failure mode to plan for is a **floor**, not a ceiling: below the gate,
CDD's separation is too close to chance for any amount of additional data
to rescue (§2.4) — this is why the check happens once, in the pilot,
*before* sizing the full run, rather than being treated as a soft warning.
"""

from __future__ import annotations

import dataclasses

from qcd.analysis.auc import paired_auc_detection_limit, quantization_delta_auc
from qcd.constants import (
    CDD_GATE_ASSUMED_SEPARATION_REDUCTION,
    CDD_GATE_AUC,
    CDD_GATE_REFERENCE_N,
    CDD_GATE_REFERENCE_R,
)


@dataclasses.dataclass
class CDDGateResult:
    measured_baseline_auc: float
    threshold: float
    passed: bool
    assumed_quantization_delta_auc: float
    detection_limit_at_reference_n: float
    reason: str


def check_cdd_gate(measured_baseline_auc: float, *, threshold: float = CDD_GATE_AUC) -> CDDGateResult:
    """`measured_baseline_auc` is CDD's bf16 AUC from the
    pilot (paper §4.7 item (b))."""
    delta_auc = quantization_delta_auc(measured_baseline_auc, CDD_GATE_ASSUMED_SEPARATION_REDUCTION)
    detection_limit = paired_auc_detection_limit(CDD_GATE_REFERENCE_N, measured_baseline_auc, CDD_GATE_REFERENCE_R)
    passed = measured_baseline_auc >= threshold

    if passed:
        reason = (
            f"measured CDD baseline AUC {measured_baseline_auc:.4f} >= gate threshold "
            f"{threshold:.4f} — CDD stays in the primary analysis."
        )
    else:
        reason = (
            f"measured CDD baseline AUC {measured_baseline_auc:.4f} < gate threshold "
            f"{threshold:.4f} — drop CDD from the primary analysis; report Q1 using "
            "probability-based detectors only, and CDD's inoperability at this scale "
            "as a standalone finding (Contribution 4, §1)."
        )

    return CDDGateResult(
        measured_baseline_auc=measured_baseline_auc,
        threshold=threshold,
        passed=passed,
        assumed_quantization_delta_auc=delta_auc,
        detection_limit_at_reference_n=detection_limit,
        reason=reason,
    )
