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

    # Residual-contamination evidence from TRACER (arXiv:2605.24079), filled in
    # during step 5 (AGENTS.md §7). It is stored separately from Q1b's temporal
    # proxy label and is not a verified negative label or a prerequisite for
    # computing Q1b. None until that measurement has actually been made.
    tracer_label: float | None = None

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


class TemporalProxyLabel(enum.Enum):
    POSSIBLE_EXPOSURE = "possible-exposure"
    CLEAN_BY_MODEL_CUTOFF = "clean-by-model-cutoff"
    SHARED_CLEAN_CONTROL = "shared-clean-control"


class CorpusReferenceStatus(enum.Enum):
    CONFIRMED_MATCH = "confirmed-match"
    NO_MATCH_FOUND = "no-match-found"
    NOT_OBSERVABLE = "not-observable"


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
