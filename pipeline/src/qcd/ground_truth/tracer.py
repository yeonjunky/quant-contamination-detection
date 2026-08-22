"""Pure decision logic for the TRACER contamination-detection pipeline.

This module contains no embedding or LLM client.  It records the deterministic
parts of TRACER described in Sections 3--4 and Appendix A of arXiv:2605.24079:
embedding triage, parsing of the two forced-choice prompts, and preservation of
the final trivial-task exclusion decision.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import re


class ContaminationLabel(str, enum.Enum):
    """TRACER's four fine-grained semantic-overlap labels."""

    FI = "FI"
    NI = "NI"
    SL = "SL"
    U = "U"


class TriageRoute(str, enum.Enum):
    """Action selected by embedding-based coarse triage."""

    DIRECT_FI = "direct_fi"
    VERIFY = "verify"
    DIRECT_U = "direct_u"


@dataclasses.dataclass(frozen=True)
class TriageResult:
    similarity: float
    route: TriageRoute
    label: ContaminationLabel | None

    @property
    def requires_verification(self) -> bool:
        return self.route is TriageRoute.VERIFY


def triage_similarity(
    similarity: float,
    *,
    lower_threshold: float = 0.6,
    upper_threshold: float = 0.9,
) -> TriageResult:
    """Apply TRACER's tuned inclusive triage boundaries.

    Scores at or above 0.9 are directly FI, scores at or below 0.6 are
    directly U, and scores strictly between the boundaries require LLM
    verification.  Custom thresholds are accepted to make sensitivity checks
    explicit rather than hidden in callers.
    """
    if not all(math.isfinite(value) for value in (similarity, lower_threshold, upper_threshold)):
        raise ValueError("similarities and thresholds must be finite")
    if not 0.0 <= similarity <= 1.0:
        raise ValueError("similarity must be in [0, 1]")
    if not 0.0 <= lower_threshold < upper_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= lower < upper <= 1")

    if similarity >= upper_threshold:
        return TriageResult(similarity, TriageRoute.DIRECT_FI, ContaminationLabel.FI)
    if similarity <= lower_threshold:
        return TriageResult(similarity, TriageRoute.DIRECT_U, ContaminationLabel.U)
    return TriageResult(similarity, TriageRoute.VERIFY, None)


_VERIFICATION_RE = re.compile(r"Answer: ([ABCD])")
_ANSWER_TO_LABEL = {
    "A": ContaminationLabel.FI,
    "B": ContaminationLabel.NI,
    "C": ContaminationLabel.SL,
    "D": ContaminationLabel.U,
}


def parse_verification_response(response: str) -> ContaminationLabel:
    """Parse exactly the forced-choice format ``Answer: A|B|C|D``."""
    match = _VERIFICATION_RE.fullmatch(response)
    if match is None:
        raise ValueError("verification response must be exactly 'Answer: A', 'B', 'C', or 'D'")
    return _ANSWER_TO_LABEL[match.group(1)]


_TRIVIAL_RE = re.compile(r"Decision: (Yes|No)\nReasoning: ([^\n]+)")


def parse_trivial_response(response: str) -> bool:
    """Parse Appendix A's exact final-screening response envelope.

    ``True`` means the task is a trivial/basic helper function.  Requiring a
    non-empty, single-line reasoning field prevents permissive substring
    parsing from silently accepting malformed model output.
    """
    match = _TRIVIAL_RE.fullmatch(response)
    if match is None:
        raise ValueError(
            "trivial response must be exactly 'Decision: Yes|No' followed by "
            "a non-empty 'Reasoning: ...' line"
        )
    return match.group(1) == "Yes"


class PairStatus(str, enum.Enum):
    INCLUDED = "included"
    EXCLUDED_TRIVIAL = "excluded_trivial"


@dataclasses.dataclass(frozen=True)
class PairResult:
    """Final pair decision, retaining screening evidence after exclusion."""

    label: ContaminationLabel
    first_task_trivial: bool
    second_task_trivial: bool
    status: PairStatus

    @property
    def excluded(self) -> bool:
        return self.status is PairStatus.EXCLUDED_TRIVIAL


def finalize_pair(
    label: ContaminationLabel,
    *,
    first_task_trivial: bool,
    second_task_trivial: bool,
) -> PairResult:
    """Exclude a pair if either task is trivial, without discarding its label."""
    status = (
        PairStatus.EXCLUDED_TRIVIAL
        if first_task_trivial or second_task_trivial
        else PairStatus.INCLUDED
    )
    return PairResult(label, first_task_trivial, second_task_trivial, status)
