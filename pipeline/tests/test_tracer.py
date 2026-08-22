import pytest

from qcd.ground_truth.tracer import (
    ContaminationLabel,
    PairStatus,
    TriageRoute,
    finalize_pair,
    parse_trivial_response,
    parse_verification_response,
    triage_similarity,
)


@pytest.mark.parametrize("score", [0.9, 0.95, 1.0])
def test_triage_directly_labels_upper_region_fi(score):
    result = triage_similarity(score)
    assert result.route is TriageRoute.DIRECT_FI
    assert result.label is ContaminationLabel.FI
    assert result.requires_verification is False


@pytest.mark.parametrize("score", [0.0, 0.4, 0.6])
def test_triage_directly_labels_lower_region_unrelated(score):
    result = triage_similarity(score)
    assert result.route is TriageRoute.DIRECT_U
    assert result.label is ContaminationLabel.U
    assert result.requires_verification is False


@pytest.mark.parametrize("score", [0.600001, 0.75, 0.899999])
def test_triage_sends_open_middle_region_to_verification(score):
    result = triage_similarity(score)
    assert result.route is TriageRoute.VERIFY
    assert result.label is None
    assert result.requires_verification is True


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf")])
def test_triage_rejects_invalid_similarity(score):
    with pytest.raises(ValueError):
        triage_similarity(score)


@pytest.mark.parametrize(
    ("response", "label"),
    [
        ("Answer: A", ContaminationLabel.FI),
        ("Answer: B", ContaminationLabel.NI),
        ("Answer: C", ContaminationLabel.SL),
        ("Answer: D", ContaminationLabel.U),
    ],
)
def test_verification_parser_maps_forced_choices(response, label):
    assert parse_verification_response(response) is label


@pytest.mark.parametrize(
    "response",
    ["A", "Answer:A", "Answer: E", "answer: A", "Answer: A\n", "Reasoning\nAnswer: A"],
)
def test_verification_parser_rejects_non_exact_output(response):
    with pytest.raises(ValueError):
        parse_verification_response(response)


def test_trivial_parser_accepts_only_prompt_envelope():
    assert parse_trivial_response("Decision: Yes\nReasoning: It is an atomic built-in operation.") is True
    assert parse_trivial_response("Decision: No\nReasoning: It requires multiple algorithmic steps.") is False


@pytest.mark.parametrize(
    "response",
    [
        "Yes",
        "Decision: yes\nReasoning: atomic",
        "Decision: Yes",
        "Decision: No\nReasoning: ",
        "Decision: No\nReasoning: valid\nextra",
    ],
)
def test_trivial_parser_rejects_malformed_output(response):
    with pytest.raises(ValueError):
        parse_trivial_response(response)


@pytest.mark.parametrize(
    ("first", "second"),
    [(True, False), (False, True), (True, True)],
)
def test_final_screening_preserves_evidence_when_either_task_is_trivial(first, second):
    result = finalize_pair(
        ContaminationLabel.NI,
        first_task_trivial=first,
        second_task_trivial=second,
    )
    assert result.status is PairStatus.EXCLUDED_TRIVIAL
    assert result.excluded is True
    assert result.label is ContaminationLabel.NI
    assert result.first_task_trivial is first
    assert result.second_task_trivial is second


def test_final_screening_includes_nontrivial_pair():
    result = finalize_pair(
        ContaminationLabel.SL,
        first_task_trivial=False,
        second_task_trivial=False,
    )
    assert result.status is PairStatus.INCLUDED
    assert result.excluded is False
