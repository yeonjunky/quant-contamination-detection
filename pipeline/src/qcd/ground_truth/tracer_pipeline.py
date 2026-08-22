"""Orchestration for one candidate pair through the TRACER stages."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from typing import Protocol

from qcd.ground_truth.tracer import (
    ContaminationLabel,
    finalize_pair,
    parse_trivial_response,
    parse_verification_response,
    triage_similarity,
)
from qcd.ground_truth.tracer_prompts import (
    render_normalization_prompt,
    render_trivial_screening_prompt,
    render_verification_prompt,
)
from qcd.ground_truth.tracer_schema import (
    DecodingConfig,
    ModelIdentity,
    TracerPairRecord,
    TrivialScreen,
)


class EmbeddingScorer(Protocol):
    def pairwise_scores(self, benchmark_descriptions, training_descriptions): ...


@dataclasses.dataclass(frozen=True)
class CandidatePair:
    benchmark: str
    item_id: str
    corpus: str
    corpus_revision: str
    shard: str
    document_id: str
    benchmark_description: str
    training_description: str
    retrieval_method: str
    retrieval_score: float


@dataclasses.dataclass(frozen=True)
class TracerRunConfig:
    run_id: str
    embedding_model: ModelIdentity
    normalizer_model: ModelIdentity
    verifier_model: ModelIdentity
    screener_model: ModelIdentity
    prompt_version: str
    decoding_config: DecodingConfig
    run_timestamp: str
    lower_threshold: float = 0.6
    upper_threshold: float = 0.9


class NormalizationCache:
    """Run-local exact-description cache with explicit LLM injection."""

    def __init__(self, generate: Callable[[str], str]) -> None:
        self._generate = generate
        self._values: dict[str, str] = {}

    def normalize(self, description: str) -> str:
        if description not in self._values:
            response = self._generate(render_normalization_prompt(description))
            if not isinstance(response, str) or not response.strip():
                raise ValueError("normalizer returned an empty or non-string response")
            self._values[description] = response.strip()
        return self._values[description]

    def __len__(self) -> int:
        return len(self._values)


class TracerPipeline:
    """Execute paper-specified routing while keeping model clients replaceable."""

    def __init__(
        self,
        *,
        config: TracerRunConfig,
        embedding_scorer: EmbeddingScorer,
        normalize: Callable[[str], str],
        verify: Callable[[str], str],
        screen: Callable[[str], str],
    ) -> None:
        self.config = config
        self.embedding_scorer = embedding_scorer
        self.normalization_cache = NormalizationCache(normalize)
        self._verify = verify
        self._screen = screen

    def run_pair(self, candidate: CandidatePair) -> TracerPairRecord:
        benchmark_normalized = self.normalization_cache.normalize(candidate.benchmark_description)
        training_normalized = self.normalization_cache.normalize(candidate.training_description)
        scores = self.embedding_scorer.pairwise_scores(
            [benchmark_normalized], [training_normalized],
        )
        if getattr(scores, "shape", None) != (1, 1):
            raise ValueError("embedding scorer must return a 1x1 matrix for one pair")
        score = float(scores[0, 0])
        triage = triage_similarity(
            score,
            lower_threshold=self.config.lower_threshold,
            upper_threshold=self.config.upper_threshold,
        )

        verification_output = None
        label = triage.label
        if triage.requires_verification:
            verification_output = self._verify(
                render_verification_prompt(benchmark_normalized, training_normalized)
            )
            label = parse_verification_response(verification_output)
        assert label is not None

        benchmark_screen = TrivialScreen(None, None)
        training_screen = TrivialScreen(None, None)
        excluded = False
        if label is not ContaminationLabel.U:
            benchmark_raw = self._screen(render_trivial_screening_prompt(benchmark_normalized))
            training_raw = self._screen(render_trivial_screening_prompt(training_normalized))
            benchmark_trivial = parse_trivial_response(benchmark_raw)
            training_trivial = parse_trivial_response(training_raw)
            benchmark_screen = TrivialScreen(benchmark_raw, benchmark_trivial)
            training_screen = TrivialScreen(training_raw, training_trivial)
            excluded = finalize_pair(
                label,
                first_task_trivial=benchmark_trivial,
                second_task_trivial=training_trivial,
            ).excluded

        return TracerPairRecord(
            run_id=self.config.run_id,
            benchmark=candidate.benchmark,
            item_id=candidate.item_id,
            corpus=candidate.corpus,
            corpus_revision=candidate.corpus_revision,
            shard=candidate.shard,
            document_id=candidate.document_id,
            benchmark_description=candidate.benchmark_description,
            training_description=candidate.training_description,
            normalized_benchmark_description=benchmark_normalized,
            normalized_training_description=training_normalized,
            retrieval_method=candidate.retrieval_method,
            retrieval_score=candidate.retrieval_score,
            embedding_model=self.config.embedding_model,
            embedding_score=score,
            triage_route=triage.route,
            raw_verification_output=verification_output,
            pre_screen_label=label,
            benchmark_trivial_screen=benchmark_screen,
            training_trivial_screen=training_screen,
            excluded=excluded,
            final_fine_grained_label=label,
            binary_contamination_label=None if excluded else label is not ContaminationLabel.U,
            normalizer_model=self.config.normalizer_model,
            verifier_model=self.config.verifier_model,
            screener_model=self.config.screener_model,
            prompt_version=self.config.prompt_version,
            decoding_config=self.config.decoding_config,
            run_timestamp=self.config.run_timestamp,
        )

    def run(self, candidates: Iterable[CandidatePair]) -> list[TracerPairRecord]:
        return [self.run_pair(candidate) for candidate in candidates]
