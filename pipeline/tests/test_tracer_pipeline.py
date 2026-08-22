import numpy as np
import pytest

from qcd.ground_truth.tracer import ContaminationLabel, TriageRoute
from qcd.ground_truth.tracer_pipeline import CandidatePair, TracerPipeline, TracerRunConfig
from qcd.ground_truth.tracer_prompts import TRACER_PROMPT_VERSION
from qcd.ground_truth.tracer_schema import (
    DecodingConfig,
    ModelIdentity,
    read_jsonl,
    write_jsonl_atomic,
)


class FixedScorer:
    def __init__(self, score):
        self.score = score
        self.calls = []

    def pairwise_scores(self, benchmark, training):
        self.calls.append((benchmark, training))
        return np.array([[self.score]])


def candidate(**changes):
    values = dict(
        benchmark="humaneval", item_id="HumanEval/0", corpus="dolma3",
        corpus_revision="corpus-rev", shard="part-0", document_id="doc-1",
        benchmark_description="benchmark original", training_description="training original",
        retrieval_method="ngram", retrieval_score=0.8,
    )
    values.update(changes)
    return CandidatePair(**values)


def config():
    model = ModelIdentity("mock/model", "model-rev")
    return TracerRunConfig(
        run_id="run-1", embedding_model=model, normalizer_model=model,
        verifier_model=model, screener_model=model,
        prompt_version=TRACER_PROMPT_VERSION,
        decoding_config=DecodingConfig(0.0, 1.0, 7, 256),
        run_timestamp="2026-08-22T00:00:00Z",
    )


def make_pipeline(score, *, verify_output="Answer: B", screens=(False, False)):
    normalize_calls = []
    verify_calls = []
    screen_calls = []
    screen_values = iter(screens)

    def normalize(prompt):
        normalize_calls.append(prompt)
        return "normalized " + ("benchmark" if "benchmark original" in prompt else "training")

    def verify(prompt):
        verify_calls.append(prompt)
        return verify_output

    def screen(prompt):
        screen_calls.append(prompt)
        decision = next(screen_values)
        return f"Decision: {'Yes' if decision else 'No'}\nReasoning: deterministic mock reason."

    pipeline = TracerPipeline(
        config=config(), embedding_scorer=FixedScorer(score),
        normalize=normalize, verify=verify, screen=screen,
    )
    return pipeline, normalize_calls, verify_calls, screen_calls


def test_low_route_skips_verification_and_screening():
    pipeline, normalizations, verifications, screens = make_pipeline(0.6)
    result = pipeline.run_pair(candidate())
    assert result.triage_route is TriageRoute.DIRECT_U
    assert result.pre_screen_label is ContaminationLabel.U
    assert result.binary_contamination_label is False
    assert result.benchmark_trivial_screen.is_trivial is None
    assert len(normalizations) == 2
    assert verifications == []
    assert screens == []


def test_ambiguous_route_verifies_and_preserves_trivial_exclusion():
    pipeline, _, verifications, screens = make_pipeline(0.75, screens=(True, False))
    result = pipeline.run_pair(candidate())
    assert result.triage_route is TriageRoute.VERIFY
    assert result.pre_screen_label is ContaminationLabel.NI
    assert result.raw_verification_output == "Answer: B"
    assert result.excluded is True
    assert result.binary_contamination_label is None
    assert len(verifications) == 1
    assert len(screens) == 2


def test_high_route_skips_verification_but_screens_both_tasks():
    pipeline, _, verifications, screens = make_pipeline(0.9)
    result = pipeline.run_pair(candidate())
    assert result.triage_route is TriageRoute.DIRECT_FI
    assert result.pre_screen_label is ContaminationLabel.FI
    assert result.binary_contamination_label is True
    assert verifications == []
    assert len(screens) == 2


def test_normalization_cache_deduplicates_exact_descriptions_across_pairs():
    pipeline, normalizations, _, _ = make_pipeline(0.0)
    pair = candidate(benchmark_description="shared", training_description="shared")
    pipeline.run([pair, pair])
    assert len(normalizations) == 1
    assert len(pipeline.normalization_cache) == 1


@pytest.mark.parametrize("response", ["", "   ", None])
def test_invalid_normalization_fails_before_embedding(response):
    scorer = FixedScorer(0.5)
    pipeline = TracerPipeline(
        config=config(), embedding_scorer=scorer, normalize=lambda _: response,
        verify=lambda _: "Answer: D", screen=lambda _: "",
    )
    with pytest.raises(ValueError, match="normalizer"):
        pipeline.run_pair(candidate())
    assert scorer.calls == []


def test_embedding_scorer_must_return_one_score_per_pair():
    class BadScorer:
        def pairwise_scores(self, benchmark, training):
            return np.array([0.5])

    pipeline = TracerPipeline(
        config=config(), embedding_scorer=BadScorer(), normalize=lambda _: "normalized",
        verify=lambda _: "Answer: D", screen=lambda _: "",
    )
    with pytest.raises(ValueError, match="1x1"):
        pipeline.run_pair(candidate())


def test_mock_pipeline_record_round_trips_through_auditable_jsonl(tmp_path):
    pipeline, _, _, _ = make_pipeline(0.75, verify_output="Answer: C")
    [record] = pipeline.run([candidate()])
    output = tmp_path / "tracer_pairs.jsonl"
    write_jsonl_atomic([record], output)
    assert list(read_jsonl(output)) == [record]
