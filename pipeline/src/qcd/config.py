"""Run configuration dataclasses shared by the mock dry-run harness and the
real H100 drivers (scripts/run_pilot.py and scripts/run_main.py). Kept
deliberately thin: this is wiring, not experimental design —
the actual axes (which models, which quant levels, which datasets) are fixed
by paper/paper_draft.md §4.1-§4.3 and mirrored in models/registry.py.
"""

from __future__ import annotations

import dataclasses
import enum

from qcd.data.schema import Dataset


class Quant(enum.Enum):
    """The four-level quantization ladder (paper §4.3).

    ``GPTQ_AWQ_INT4`` is retained as a historical storage/API value; every
    current arm implements that rung with AWQ.
    """

    BF16 = "bf16"
    BNB_INT8 = "bnb_int8"
    BNB_NF4 = "bnb_nf4"
    GPTQ_AWQ_INT4 = "gptq_awq_int4"


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    name: str  # paper's own model label, e.g. "Qwen2.5-7B"
    param_count_b: float
    # Verified Hugging Face repository ID; revision pins the exact snapshot.
    hf_repo_id: str
    revision: str  # immutable Hugging Face commit SHA
    role: str  # paper §4.1's "Role" column, e.g. "Pilot workhorse + size axis"
    included_in_main_analysis: bool = True


@dataclasses.dataclass(frozen=True)
class QuantSpec:
    quant: Quant
    # Loader backend implementing this level.
    backend: str


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    dataset: Dataset
    target_n: int
    release_version: str | None = None


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """One (model, quant, dataset) cell's generation/scoring configuration."""

    model: ModelSpec
    quant: QuantSpec
    dataset: DatasetSpec
    n_cdd_samples: int
    cdd_sample_temperature: float
    seed: int
    mock: bool = False
