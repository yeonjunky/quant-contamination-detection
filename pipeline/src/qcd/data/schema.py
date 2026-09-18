"""Canonical per-item schema shared by dataset loaders, models/mock.py's synthetic
harness, and io/raw_writer.py.

Every dataset loader (data/livecodebench.py, data/humaneval.py, data/mbppplus.py,
not yet written) must produce this same shape, so nothing downstream branches on
which of the four conditions an item came from.

Field choices track the paper draft's own condition table (paper/paper_draft.md,
"Datasets and conditions"), not a re-derivation:
  - LiveCodeBench pre-cutoff = primary contamination-suspect condition
  - LiveCodeBench post-cutoff = primary clean control
  - HumanEval = secondary contamination-suspect condition (164-item hard ceiling)
  - MBPP+ = secondary contamination-suspect condition, a separate arm — never
    pooled with HumanEval (different difficulty distributions would reintroduce
    the base-rate confound inside a nominally single condition; see
    analysis/aggregation.py's pooling guard, not yet written)
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json


class Dataset(enum.Enum):
    LCB_PRE = "lcb_pre"
    LCB_POST = "lcb_post"
    HUMANEVAL = "humaneval"
    MBPPPLUS = "mbppplus"


# Legacy dataset-level grouping used by the mock harness and secondary
# HumanEval/MBPP+ contrasts. Primary LCB analyses must use the separately
# materialized model–item temporal labels, not this coarse property.
_PROXY_CONTAMINATED = {
    Dataset.LCB_PRE: True,
    Dataset.LCB_POST: False,
    Dataset.HUMANEVAL: True,
    Dataset.MBPPPLUS: True,
}


@dataclasses.dataclass
class Item:
    item_id: str
    dataset: Dataset
    prompt: str

    # Difficulty bucket, when the source dataset reports one (e.g. LiveCodeBench's
    # easy/medium/hard tags). None where no native difficulty field exists —
    # analysis/logodds.py's difficulty-stratification check (paper §4.5.3) may
    # derive an empirical bucket separately rather than relying on this being set.
    difficulty: str | None = None

    # Corpus-reference evidence (paper §4.2's separate, non-exclusive axis).
    # One entry per (model, method family, corpus stage) cell that has actually
    # been searched — see `CorpusReferenceRecord`. Empty until step 5 has run;
    # an empty tuple means "not measured", which is not the same as
    # `not-observable` (a recorded statement that the corpus cannot be
    # searched at all).
    corpus_reference: tuple["CorpusReferenceRecord", ...] = ()

    # Dataset snapshot/version pin (e.g. LiveCodeBench release_version, or the
    # evalplus version for HumanEval/MBPP+) — required for reproducibility;
    # see pipeline_build_plan.md's "open assumptions" #3-4.
    release_version: str | None = None

    # Escape hatch for dataset-specific fields (e.g. LiveCodeBench's original
    # difficulty tag string, contest date) that don't belong in the shared schema.
    metadata: dict = dataclasses.field(default_factory=dict)

    @property
    def contamination_proxy(self) -> bool:
        """Coarse dataset-level exposure proxy for legacy/secondary paths.

        Primary LCB Q1b/Q2 analyses use `model_item_labels.parquet`; this
        property must not be used as a model-specific training-cutoff label.
        """
        return _PROXY_CONTAMINATED[self.dataset]

    def corpus_status(self, model: str, family: "CorpusEvidenceFamily | str") -> "CorpusReferenceStatus | None":
        """The recorded status for one (model, method family) cell, or None
        when that cell has not been searched.

        Deliberately keyed by model: paper §4.2 stores corpus evidence on a
        per-model axis, and the two open-corpus arms do not share a
        pretraining corpus (`ground_truth/olmo_corpora.py`).
        """
        family_value = family.value if isinstance(family, CorpusEvidenceFamily) else family
        statuses = {
            record.status
            for record in self.corpus_reference
            if record.model == model and record.method_family == family_value
        }
        if not statuses:
            return None
        if CorpusReferenceStatus.CONFIRMED_MATCH in statuses:
            # Non-exclusive axis: a confirmed match in any searched stage is a
            # confirmed match for that (model, family) cell.
            return CorpusReferenceStatus.CONFIRMED_MATCH
        if statuses == {CorpusReferenceStatus.NO_MATCH_FOUND}:
            return CorpusReferenceStatus.NO_MATCH_FOUND
        return CorpusReferenceStatus.NOT_OBSERVABLE


class TemporalProxyLabel(enum.Enum):
    POSSIBLE_EXPOSURE = "possible-exposure"
    CLEAN_BY_MODEL_CUTOFF = "clean-by-model-cutoff"
    SHARED_CLEAN_CONTROL = "shared-clean-control"


class CorpusReferenceStatus(enum.Enum):
    CONFIRMED_MATCH = "confirmed-match"
    NO_MATCH_FOUND = "no-match-found"
    NOT_OBSERVABLE = "not-observable"


class CorpusEvidenceFamily(enum.Enum):
    """Paper §5 step 5's three open-data detection families, which are
    reported separately: "Report `confirmed-match`, `no-match-found`, and
    `not-observable` counts from each family separately."

    TRACER is deliberately *not* a member. §5 step 5 runs TRACER alongside
    these families and then compares "the TRACER reimplementation's output
    with confirmed positive matches descriptively" — its output is the
    fine-grained `FI/NI/SL/U` label of `ground_truth/tracer_schema.py`, on its
    own record, not a fourth three-state corpus status. Mapping one onto the
    other is not specified by the paper and is not invented here.
    """

    INSTANCE_STRING = "instance_string"      # (i) exact / near-exact n-gram overlap
    SURFACE_PROGRAM = "surface_program"      # (ii) edit distance + AST similarity
    PARAPHRASE = "paraphrase"                # (iii) embedding retrieval + LLM judge


CORPUS_EVIDENCE_FAMILY_VALUES: tuple[str, ...] = tuple(
    family.value for family in CorpusEvidenceFamily
)


@dataclasses.dataclass(frozen=True)
class CorpusReferenceRecord:
    """One searched (model, method family, corpus stage) cell for one item.

    Paper §4.2: "Corpus evidence is stored on a **separate, non-exclusive
    axis** — `confirmed-match`, `no-match-found`, or `not-observable` — rather
    than collapsed into the temporal label. In particular, `no-match-found` is
    not renamed `clean`."

    `model` is the `models/registry.py` name, because the corpus that can be
    searched is a property of the checkpoint, not of the item: Qwen2.5 and
    Llama-3.1 are `not-observable` throughout, and the two Olmo arms have
    different pretraining corpora.
    """

    model: str
    method_family: str
    status: CorpusReferenceStatus
    corpus: str | None = None
    corpus_revision: str | None = None
    stage: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty registry model name")
        if self.method_family not in CORPUS_EVIDENCE_FAMILY_VALUES:
            raise ValueError(
                f"method_family must be one of {CORPUS_EVIDENCE_FAMILY_VALUES}, "
                f"got {self.method_family!r} (paper §5 step 5's three families)"
            )
        if not isinstance(self.status, CorpusReferenceStatus):
            raise ValueError("status must be a CorpusReferenceStatus")


def corpus_reference_to_json(records: "tuple[CorpusReferenceRecord, ...]") -> str:
    """Serialize the axis for the `corpus_reference_json` parquet column."""
    return json.dumps(
        [
            {
                "model": record.model,
                "method_family": record.method_family,
                "status": record.status.value,
                "corpus": record.corpus,
                "corpus_revision": record.corpus_revision,
                "stage": record.stage,
            }
            for record in records
        ],
        sort_keys=True,
    )


def corpus_reference_from_json(value: object) -> tuple[CorpusReferenceRecord, ...]:
    """Read the axis back, tolerating a parquet file written before it existed.

    A row from an older `items.parquet` has no `corpus_reference_json` column
    at all (pandas yields None/NaN for the missing value), and older files
    carry a `tracer_label` float column instead. Both read back as "not
    measured" rather than raising, so an existing raw tree still loads.
    """
    if value is None or isinstance(value, float):  # None, or pandas' NaN
        return ()
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        if not value.strip():
            return ()
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError(f"corpus_reference_json must hold a JSON list, got {type(value).__name__}")
    return tuple(
        CorpusReferenceRecord(
            model=entry["model"],
            method_family=entry["method_family"],
            status=CorpusReferenceStatus(entry["status"]),
            corpus=entry.get("corpus"),
            corpus_revision=entry.get("corpus_revision"),
            stage=entry.get("stage"),
        )
        for entry in value
    )


@dataclasses.dataclass
class PromptScoringDetail:
    """One fixed-prompt teacher-forced scoring pass, with the provenance
    paper §4.4 requires alongside the numbers themselves: "Record target text,
    token boundaries, truncation, chat template, tokenizer/checkpoint
    revisions and decoding settings so precision comparisons use identical
    text and scoring rules."

    Produced by `models/loader.py`'s `_RealModelAdapter.score_prompt_detail()`
    and written out by `real_run.py` through `io/raw_writer.py`. The revisions
    and decoding settings named in that sentence are recorded separately (on
    the same generations row, and in the run manifest respectively), because
    they are properties of the loaded model and of the run, not of one
    scoring pass.

    - `logprobs` — per-token natural-log probabilities, one per scored token,
      in order. This is exactly what `score_prompt_logprobs()` returns.
    - `target_token_indices` — the positions, in the rendered (chat-templated)
      token sequence, whose log-probability entered `logprobs`. Template,
      special and generation-marker tokens are absent by construction, as is
      position 0 (no causal left context).
    - `target_char_span` — (start, end) character offsets of the benchmark
      text inside the rendered string.
    - `rendered_char_length` — length of the rendered string, so a stored span
      can be interpreted without re-rendering.
    - `chat_template` — the tokenizer's own chat-template source, or None when
      the tokenizer has none (the plain-tokenization fallback path).
    """

    logprobs: list[float]
    target_token_indices: list[int]
    target_char_span: tuple[int, int]
    rendered_char_length: int
    chat_template_applied: bool
    chat_template: str | None
    target_text: str

    @property
    def target_text_sha256(self) -> str:
        """Identity of the scored text itself — lets a later check confirm
        that every precision scored byte-identical text without re-reading
        items.parquet."""
        return hashlib.sha256(self.target_text.encode()).hexdigest()

    @property
    def chat_template_id(self) -> str:
        """Content address of the rendered chat template. The template is
        identical for every item of a given tokenizer, so generations rows
        store this id and the template text itself is stored once per run
        (`chat_templates.parquet`)."""
        return hashlib.sha256((self.chat_template or "").encode()).hexdigest()[:16]
