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


class Dataset(enum.Enum):
    LCB_PRE = "lcb_pre"
    LCB_POST = "lcb_post"
    HUMANEVAL = "humaneval"
    MBPPPLUS = "mbppplus"


# Dataset-design contamination label: True for the two contamination-suspect
# conditions, False for the one clean control. This is a coarse *proxy* label
# with a nonzero error rate e (paper §4.5.2) — TRACER (Item.tracer_label)
# measures the residual contamination that this proxy misses, it does not
# replace it.
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

    # Residual contamination score from TRACER (arXiv:2605.24079), filled in
    # during step 5 (CLAUDE.md §7) — a prerequisite for Q1b, not for Q1a.
    # None until that measurement has actually been made; do not default this
    # to a placeholder number.
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
        """Dataset-design contamination-suspect label (not TRACER's measured
        residual). True for LCB-pre/HumanEval/MBPP+, False for LCB-post."""
        return _PROXY_CONTAMINATED[self.dataset]
