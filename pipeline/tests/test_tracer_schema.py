import dataclasses

import pytest

from qcd.ground_truth.tracer import ContaminationLabel, TriageRoute
from qcd.ground_truth.tracer_schema import (
    CoverageStatus,
    DecodingConfig,
    ItemCoverageRecord,
    ModelIdentity,
    TracerPairRecord,
    TrivialScreen,
    read_jsonl,
    write_jsonl_atomic,
)


def pair(label=ContaminationLabel.FI, *, excluded=False):
    model = ModelIdentity("test/model", "abc123")
    trivial = TrivialScreen("Decision: Yes\nReasoning: basic", True) if excluded else TrivialScreen(
        "Decision: No\nReasoning: nontrivial", False
    )
    return TracerPairRecord(
        run_id="run-1", benchmark="humaneval", item_id="HumanEval/0",
        corpus="dolma3", corpus_revision="rev", shard="part-0", document_id="doc-1",
        benchmark_description="original benchmark", training_description="original training",
        normalized_benchmark_description="normalized benchmark",
        normalized_training_description="normalized training", retrieval_method="ngram",
        retrieval_score=0.8, embedding_model=model, embedding_score=0.95,
        triage_route=TriageRoute.DIRECT_FI, raw_verification_output=None,
        pre_screen_label=label, benchmark_trivial_screen=trivial,
        training_trivial_screen=TrivialScreen("Decision: No\nReasoning: nontrivial", False),
        excluded=excluded, final_fine_grained_label=label,
        binary_contamination_label=None if excluded else label is not ContaminationLabel.U,
        normalizer_model=model, verifier_model=model, screener_model=model,
        prompt_version="tracer-paper-appendix-a-v1",
        decoding_config=DecodingConfig(0.0, 1.0, 42, 256, {"retry_limit": 1}),
        run_timestamp="2026-08-22T00:00:00Z",
    )


@pytest.mark.parametrize("label", [ContaminationLabel.FI, ContaminationLabel.NI, ContaminationLabel.SL])
def test_included_positive_labels_map_to_true(label):
    record = pair()
    route = TriageRoute.DIRECT_FI if label is ContaminationLabel.FI else TriageRoute.VERIFY
    answer = {ContaminationLabel.NI: "B", ContaminationLabel.SL: "C"}.get(label)
    record = dataclasses.replace(
        record, pre_screen_label=label, final_fine_grained_label=label,
        triage_route=route,
        raw_verification_output=(
            None if route is TriageRoute.DIRECT_FI else f"Answer: {answer}"
        ),
    )
    assert record.binary_contamination_label is True


def test_included_u_maps_to_false():
    record = dataclasses.replace(
        pair(), pre_screen_label=ContaminationLabel.U,
        final_fine_grained_label=ContaminationLabel.U, binary_contamination_label=False,
        triage_route=TriageRoute.DIRECT_U,
    )
    assert record.binary_contamination_label is False


def test_excluded_pair_requires_null_binary_label():
    assert pair(excluded=True).binary_contamination_label is None
    with pytest.raises(ValueError, match="map to null"):
        dataclasses.replace(pair(excluded=True), binary_contamination_label=True)


def test_screening_cannot_rewrite_semantic_label():
    with pytest.raises(ValueError, match="must not rewrite"):
        dataclasses.replace(pair(), final_fine_grained_label=ContaminationLabel.NI)


def test_raw_outputs_must_match_their_parsed_fields():
    with pytest.raises(ValueError, match="verification output"):
        dataclasses.replace(
            pair(), triage_route=TriageRoute.VERIFY,
            raw_verification_output="Answer: D",
        )
    with pytest.raises(ValueError, match="trivial-screen output"):
        dataclasses.replace(
            pair(), benchmark_trivial_screen=TrivialScreen(
                "Decision: Yes\nReasoning: atomic", False,
            ),
        )


def test_coverage_distinguishes_unsearched_and_completed_empty():
    common = dict(
        run_id="run-1", benchmark="mbpp", item_id="1", corpus="dolma3",
        corpus_revision="rev", retrieval_method="ngram", retrieval_config={"n": 13},
        expected_shards=10, run_timestamp="2026-08-22T00:00:00Z",
    )
    unsearched = ItemCoverageRecord(
        **common, status=CoverageStatus.NOT_SEARCHED, candidate_count=None, searched_shards=0
    )
    empty = ItemCoverageRecord(
        **common, status=CoverageStatus.SEARCH_COMPLETE_NO_CANDIDATES,
        candidate_count=0, searched_shards=10,
    )
    assert unsearched.status is not empty.status
    with pytest.raises(ValueError, match="candidate_count=0"):
        dataclasses.replace(empty, candidate_count=1)


def test_atomic_jsonl_round_trip(tmp_path):
    coverage = ItemCoverageRecord(
        run_id="run-1", benchmark="humaneval", item_id="0", corpus="dolma3",
        corpus_revision="rev", retrieval_method="ngram", retrieval_config={"n": 13},
        status=CoverageStatus.SEARCH_COMPLETE_WITH_CANDIDATES, candidate_count=1,
        expected_shards=1, searched_shards=1, run_timestamp="2026-08-22T00:00:00Z",
    )
    destination = tmp_path / "nested" / "records.jsonl"
    write_jsonl_atomic([pair(), coverage], destination)
    assert list(read_jsonl(destination)) == [pair(), coverage]


def test_failed_atomic_write_preserves_existing_file(tmp_path):
    destination = tmp_path / "records.jsonl"
    destination.write_text("original\n", encoding="utf-8")
    with pytest.raises(TypeError):
        write_jsonl_atomic([pair(), object()], destination)
    assert destination.read_text(encoding="utf-8") == "original\n"
