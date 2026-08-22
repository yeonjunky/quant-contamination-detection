"""Auditable JSONL records for the TRACER reimplementation.

The pair record deliberately keeps retrieval, routing, raw model outputs, and
the post-screen decision together.  The coverage record prevents an item for
which retrieval was never run from being confused with a completed search that
returned no candidates.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from qcd.ground_truth.tracer import (
    ContaminationLabel,
    TriageRoute,
    parse_trivial_response,
    parse_verification_response,
)


class CoverageStatus(str, enum.Enum):
    NOT_SEARCHED = "not_searched"
    SEARCH_COMPLETE_NO_CANDIDATES = "search_complete_no_candidates"
    SEARCH_COMPLETE_WITH_CANDIDATES = "search_complete_with_candidates"


@dataclasses.dataclass(frozen=True)
class ModelIdentity:
    model: str
    revision: str

    def __post_init__(self) -> None:
        _require_text("model", self.model)
        _require_text("revision", self.revision)


@dataclasses.dataclass(frozen=True)
class DecodingConfig:
    temperature: float
    top_p: float
    seed: int | None
    max_output_tokens: int
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")


@dataclasses.dataclass(frozen=True)
class TrivialScreen:
    raw_output: str | None
    is_trivial: bool | None

    def __post_init__(self) -> None:
        if (self.raw_output is None) != (self.is_trivial is None):
            raise ValueError("raw trivial-screen output and parsed decision must both be null or set")


@dataclasses.dataclass(frozen=True)
class TracerPairRecord:
    run_id: str
    benchmark: str
    item_id: str
    corpus: str
    corpus_revision: str
    shard: str
    document_id: str
    benchmark_description: str
    training_description: str
    normalized_benchmark_description: str
    normalized_training_description: str
    retrieval_method: str
    retrieval_score: float
    embedding_model: ModelIdentity
    embedding_score: float
    triage_route: TriageRoute
    raw_verification_output: str | None
    pre_screen_label: ContaminationLabel
    benchmark_trivial_screen: TrivialScreen
    training_trivial_screen: TrivialScreen
    excluded: bool
    final_fine_grained_label: ContaminationLabel
    binary_contamination_label: bool | None
    normalizer_model: ModelIdentity
    verifier_model: ModelIdentity
    screener_model: ModelIdentity
    prompt_version: str
    decoding_config: DecodingConfig
    run_timestamp: str
    record_type: str = dataclasses.field(default="pair", init=False)

    def __post_init__(self) -> None:
        for name in (
            "run_id", "benchmark", "item_id", "corpus", "corpus_revision", "shard",
            "document_id", "benchmark_description", "training_description",
            "normalized_benchmark_description", "normalized_training_description",
            "retrieval_method", "prompt_version", "run_timestamp",
        ):
            _require_text(name, getattr(self, name))
        for name in ("retrieval_score", "embedding_score"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.embedding_score <= 1:
            raise ValueError("embedding_score must be in [0, 1]")
        if self.final_fine_grained_label is not self.pre_screen_label:
            raise ValueError("trivial screening must not rewrite the pre-screen semantic label")

        decisions = (
            self.benchmark_trivial_screen.is_trivial,
            self.training_trivial_screen.is_trivial,
        )
        has_trivial = any(value is True for value in decisions)
        if self.excluded != has_trivial:
            raise ValueError("excluded must be true exactly when either task is marked trivial")
        expected = None if self.excluded else self.pre_screen_label is not ContaminationLabel.U
        if self.binary_contamination_label is not expected:
            raise ValueError("excluded pairs map to null; FI/NI/SL map to true; U maps to false")

        if self.triage_route is TriageRoute.DIRECT_FI:
            if self.pre_screen_label is not ContaminationLabel.FI or self.raw_verification_output is not None:
                raise ValueError("direct_fi requires FI and no verification output")
        elif self.triage_route is TriageRoute.DIRECT_U:
            if self.pre_screen_label is not ContaminationLabel.U or self.raw_verification_output is not None:
                raise ValueError("direct_u requires U and no verification output")
        elif self.raw_verification_output is None:
            raise ValueError("verify route requires a raw verification output")
        elif parse_verification_response(self.raw_verification_output) is not self.pre_screen_label:
            raise ValueError("parsed verification output does not match pre_screen_label")

        for screen in (self.benchmark_trivial_screen, self.training_trivial_screen):
            if screen.raw_output is not None:
                if parse_trivial_response(screen.raw_output) is not screen.is_trivial:
                    raise ValueError("parsed trivial-screen output does not match is_trivial")


@dataclasses.dataclass(frozen=True)
class ItemCoverageRecord:
    run_id: str
    benchmark: str
    item_id: str
    corpus: str
    corpus_revision: str
    retrieval_method: str
    retrieval_config: dict[str, Any]
    status: CoverageStatus
    candidate_count: int | None
    expected_shards: int
    searched_shards: int
    run_timestamp: str
    record_type: str = dataclasses.field(default="item_coverage", init=False)

    def __post_init__(self) -> None:
        for name in (
            "run_id", "benchmark", "item_id", "corpus", "corpus_revision",
            "retrieval_method", "run_timestamp",
        ):
            _require_text(name, getattr(self, name))
        if self.expected_shards < 0 or not 0 <= self.searched_shards <= self.expected_shards:
            raise ValueError("shard counts must satisfy 0 <= searched <= expected")
        if self.status is CoverageStatus.NOT_SEARCHED:
            if self.candidate_count is not None or self.searched_shards != 0:
                raise ValueError("not_searched requires null candidate_count and zero searched shards")
            return
        if self.searched_shards != self.expected_shards:
            raise ValueError("a completed search must cover every expected shard")
        if self.status is CoverageStatus.SEARCH_COMPLETE_NO_CANDIDATES:
            if self.candidate_count != 0:
                raise ValueError("no-candidate status requires candidate_count=0")
        elif self.candidate_count is None or self.candidate_count <= 0:
            raise ValueError("with-candidates status requires a positive candidate_count")


TracerRecord = TracerPairRecord | ItemCoverageRecord


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _to_dict(record: TracerRecord) -> dict[str, Any]:
    return dataclasses.asdict(record)


def record_from_dict(data: dict[str, Any]) -> TracerRecord:
    """Construct and validate a typed record from its JSON representation."""
    values = dict(data)
    record_type = values.pop("record_type", None)
    if record_type == "item_coverage":
        values["status"] = CoverageStatus(values["status"])
        return ItemCoverageRecord(**values)
    if record_type != "pair":
        raise ValueError(f"unknown record_type: {record_type!r}")
    for field in ("embedding_model", "normalizer_model", "verifier_model", "screener_model"):
        values[field] = ModelIdentity(**values[field])
    values["decoding_config"] = DecodingConfig(**values["decoding_config"])
    for field in ("benchmark_trivial_screen", "training_trivial_screen"):
        values[field] = TrivialScreen(**values[field])
    values["triage_route"] = TriageRoute(values["triage_route"])
    values["pre_screen_label"] = ContaminationLabel(values["pre_screen_label"])
    values["final_fine_grained_label"] = ContaminationLabel(values["final_fine_grained_label"])
    return TracerPairRecord(**values)


def read_jsonl(path: str | Path) -> Iterator[TracerRecord]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record must be a JSON object")
                yield record_from_dict(value)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid record at {path}:{line_number}: {error}") from error


def write_jsonl_atomic(records: Iterable[TracerRecord], path: str | Path) -> None:
    """Replace a JSONL file atomically after every record validates/serializes."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            for record in records:
                if not isinstance(record, (TracerPairRecord, ItemCoverageRecord)):
                    raise TypeError("records must be TracerPairRecord or ItemCoverageRecord")
                json.dump(_to_dict(record), stream, ensure_ascii=False, sort_keys=True)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
