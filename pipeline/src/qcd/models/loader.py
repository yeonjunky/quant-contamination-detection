"""load_model(spec, quant) — branches fp16/bnb-int8/bnb-nf4/gptqmodel/
llm-compressor-awq/mock.

All real (non-mock) backends lazy-import their heavy dependencies inside the
branch functions, not at module scope. This lets `import qcd.models.loader`
succeed on the mock-only local profile (no torch/transformers-with-torch/
bitsandbytes installed) — only actually calling a real backend requires the
H100 GPU stack (requirements-h100.txt).
"""

from __future__ import annotations

from typing import Protocol

from qcd.config import ModelSpec, Quant
from qcd.models.mock import MockModel, MockTokenizer


class LoadedModel(Protocol):
    """The call surface every backend (real or mock) must expose — the rest
    of the pipeline (generation/sampler.py, scoring/logprob.py) is written
    against this, never against a specific backend."""

    tokenizer: object

    def generate(self, item_id: str, prompt: str, *, contaminated: bool, temperature: float, sample_id: int): ...

    def score_logprobs(self, item_id: str, token_ids: list[int], *, contaminated: bool) -> list[float]: ...


def load_model(spec: ModelSpec, quant: Quant, *, mock: bool = False) -> LoadedModel:
    if mock:
        return MockModel(MockTokenizer())

    backend = {
        Quant.FP16: _load_fp16,
        Quant.BNB_INT8: _load_bnb,
        Quant.BNB_NF4: _load_bnb,
        Quant.GPTQ_AWQ_INT4: _load_gptq_or_awq,
    }[quant]
    return backend(spec, quant)


def _load_fp16(spec: ModelSpec, quant: Quant) -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo_id)
    model = AutoModelForCausalLM.from_pretrained(spec.hf_repo_id, torch_dtype="auto", device_map="auto")
    return _RealModelAdapter(model, tokenizer)


def _load_bnb(spec: ModelSpec, quant: Quant) -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: PLC0415

    if quant is Quant.BNB_INT8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant is Quant.BNB_NF4:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    else:
        raise ValueError(f"not a bitsandbytes quant level: {quant}")

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo_id)
    model = AutoModelForCausalLM.from_pretrained(spec.hf_repo_id, quantization_config=bnb_config, device_map="auto")
    return _RealModelAdapter(model, tokenizer)


def _load_gptq_or_awq(spec: ModelSpec, quant: Quant) -> LoadedModel:
    # GPTQModel / llm-compressor, per pipeline_build_plan.md's substitution
    # for the archived AutoGPTQ/AutoAWQ. Which of the two a given model arm
    # uses is a per-model choice not yet pinned — raise loudly rather than
    # guess, per CLAUDE.md §3.1's "don't guess it" discipline.
    raise NotImplementedError(
        "GPTQModel/llm-compressor loading path not yet implemented — pin the "
        "per-model GPTQ-vs-AWQ choice and exact quantized checkpoint before "
        "wiring this up (pipeline_build_plan.md, open assumption #1)."
    )


class _RealModelAdapter:
    """Wraps a real transformers model+tokenizer to expose the same
    generate()/score_logprobs() surface as MockModel, so callers never branch
    on mock-vs-real. Not exercised by the mock-only dry run; covered by the
    real nf4 smoke test (scripts/run_smoke_test.py) instead."""

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, item_id: str, prompt: str, *, contaminated: bool, temperature: float, sample_id: int):
        raise NotImplementedError("real generation path — implement alongside scripts/run_smoke_test.py")

    def score_logprobs(self, item_id: str, token_ids: list[int], *, contaminated: bool) -> list[float]:
        raise NotImplementedError("real teacher-forced scoring path — implement alongside scripts/run_smoke_test.py")
