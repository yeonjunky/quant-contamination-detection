"""Mirrors paper/paper_draft.md §4.1's model table 1:1. If the paper's model
table changes, update this module in the same edit and rerun
tests/test_registry.py.

`hf_repo_id` values are execution identifiers, not citations:
the paper draft names models by their public label (e.g. "Qwen2.5-7B"), not
by HF hub repo ID. Every execution snapshot is pinned by immutable commit SHA.
"""

from __future__ import annotations

from qcd.config import ModelSpec

# Primary five-model comparison (paper §4.1, 2026-08-05 roster). Architecture
# is not a control axis (AGENTS.md §5 point 8: Qwen2.5 and Llama-3.1 are both
# dense GQA+RoPE); the only axis that carries information across models is
# size; Olmo additionally provides corpus transparency for ground truth.
# Llama-3.3-70B and Gemma-4-31B-it were removed from the design on
# 2026-08-05 (single-device compute ceiling / QAT confounds — see
# paper/revision_provenance.md); do not re-add them here without a matching
# paper §4.1 change.
# Instruct variants confirmed 2026-08-05 (user decision): code-generation
# pass@1 under instruction prompts is the measured quantity, §4.5.3's
# illustrative base rates are instruction-tuned figures, and the LLMLagBench
# verification (refusal-tracking Q&A probing) targets instruct checkpoints.
QWEN2_5_7B = ModelSpec(
    name="Qwen2.5-7B-Instruct",
    param_count_b=7,
    hf_repo_id="Qwen/Qwen2.5-7B-Instruct",
    revision="a09a35458c702b33eeacc393d103063234e8bc28",
    role="Pilot workhorse + size axis",
)
QWEN2_5_32B = ModelSpec(
    name="Qwen2.5-32B-Instruct",
    param_count_b=32.5,
    hf_repo_id="Qwen/Qwen2.5-32B-Instruct",
    revision="5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    role="Primary",
)
LLAMA3_1_8B = ModelSpec(
    name="Llama-3.1-8B-Instruct",
    param_count_b=8,
    hf_repo_id="meta-llama/Llama-3.1-8B-Instruct",
    revision="0e9e39f249a16976918f6564b8830bc894c89659",
    role="Size axis + externally verified cutoff (LLMLagBench: declared 2023-12, detected 2023-03)",
)
OLMO3_7B = ModelSpec(
    name="Olmo3-7B-Instruct",
    param_count_b=7,
    hf_repo_id="allenai/Olmo-3-7B-Instruct",
    revision="6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
    role="Ground-truth label validation + size axis",
)
OLMO3_1_32B = ModelSpec(
    name="Olmo3.1-32B-Instruct",
    param_count_b=32,
    hf_repo_id="allenai/Olmo-3.1-32B-Instruct",
    revision="ac0587e4a7744a551c059d8cd17ba220bc940dae",
    role="Ground-truth label validation + size axis",
)

MAIN_ANALYSIS_MODELS: tuple[ModelSpec, ...] = (
    QWEN2_5_7B,
    QWEN2_5_32B,
    LLAMA3_1_8B,
    OLMO3_7B,
    OLMO3_1_32B,
)

ALL_MODELS: tuple[ModelSpec, ...] = MAIN_ANALYSIS_MODELS

# Pilot arm (paper §4.7): Qwen2.5-7B and Olmo3-7B, BNB-nf4 first.
PILOT_MODELS: tuple[ModelSpec, ...] = (QWEN2_5_7B, OLMO3_7B)


def get_model(name: str) -> ModelSpec:
    for model in ALL_MODELS:
        if model.name == name:
            return model
    raise KeyError(f"unknown model {name!r}; known: {[m.name for m in ALL_MODELS]}")
