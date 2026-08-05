"""Mirrors paper/paper_draft.md §4.1's model table 1:1. Kept in sync manually
(CLAUDE.md has no automated cross-check for this file) — if the paper's model
table changes, update this module in the same edit and re-run
tests/test_registry.py.

`hf_repo_id` values are engineering placeholders, not verified citations:
the paper draft names models by their public label (e.g. "Qwen2.5-7B"), not
by HF hub repo id. Confirm every id against the actual hub listing (gated
status, exact revision) before the first real H100 load — see
pipeline_build_plan.md's "open assumptions".
"""

from __future__ import annotations

from qcd.config import ModelSpec

# Primary five-model comparison (paper §4.1, 2026-08-05 roster). Architecture
# is not a control axis (CLAUDE.md §5 point 8: Qwen2.5 and Llama-3.1 are both
# dense GQA+RoPE); the only axis that carries information across models is
# size. Llama-3.3-70B and Gemma-4-31B-it were removed from the design on
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
    role="Pilot workhorse + size axis",
)
QWEN2_5_32B = ModelSpec(
    name="Qwen2.5-32B-Instruct",
    param_count_b=32.5,
    hf_repo_id="Qwen/Qwen2.5-32B-Instruct",
    role="Primary",
)
LLAMA3_1_8B = ModelSpec(
    name="Llama-3.1-8B-Instruct",
    param_count_b=8,
    hf_repo_id="meta-llama/Llama-3.1-8B-Instruct",
    role="Size axis + externally verified cutoff (LLMLagBench: declared 2023-12, detected 2023-03)",
)
OLMO3_7B = ModelSpec(
    name="Olmo3-7B-Instruct",
    param_count_b=7,
    hf_repo_id="allenai/Olmo-3-7B-Instruct",
    role="Ground-truth label validation + size axis",
)
OLMO3_32B = ModelSpec(
    name="Olmo3-32B-Instruct",
    param_count_b=32,
    hf_repo_id="allenai/Olmo-3-32B-Instruct",
    role="Ground-truth label validation + size axis",
)

MAIN_ANALYSIS_MODELS: tuple[ModelSpec, ...] = (
    QWEN2_5_7B,
    QWEN2_5_32B,
    LLAMA3_1_8B,
    OLMO3_7B,
    OLMO3_32B,
)

ALL_MODELS: tuple[ModelSpec, ...] = MAIN_ANALYSIS_MODELS

# Pilot arm (paper §4.7): Qwen2.5-7B and Olmo3-7B, BNB-nf4 first.
PILOT_MODELS: tuple[ModelSpec, ...] = (QWEN2_5_7B, OLMO3_7B)


def get_model(name: str) -> ModelSpec:
    for model in ALL_MODELS:
        if model.name == name:
            return model
    raise KeyError(f"unknown model {name!r}; known: {[m.name for m in ALL_MODELS]}")
