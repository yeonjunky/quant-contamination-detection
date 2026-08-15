"""load_model(spec, quant) — branches fp16/bnb-int8/bnb-nf4/gptqmodel/
llm-compressor-awq/mock.

All real (non-mock) backends lazy-import their heavy dependencies inside the
branch functions, not at module scope. This lets `import qcd.models.loader`
succeed on the mock-only local profile (no torch/transformers-with-torch/
bitsandbytes installed) — only actually calling a real backend requires the
H100 GPU stack (requirements-h100.txt).
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Protocol

from qcd.config import ModelSpec, Quant
from qcd.models.mock import MockModel, MockTokenizer

# Engineering default, not a paper-derived constant (constants.py is scoped to
# CLAUDE.md/paper_draft.md-derived statistical/design values) — deliberately
# separate from that module.
_DEFAULT_MAX_NEW_TOKENS = 512


def _seed_from(*parts: object) -> int:
    """Deterministic 32-bit seed derived from arbitrary parts. Independent,
    smaller copy of mock.py's helper of the same name — kept un-shared on
    purpose, mirroring generation/sampler.py's ItemGenerations.greedy being
    "untyped to avoid coupling to models.mock": the real backend shouldn't
    depend on the mock module for anything."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


@dataclasses.dataclass
class _RealGenerationSample:
    """Duck-typed match for mock.GenerationSample's shape — deliberately a
    separate class, not imported from models.mock (see _seed_from's
    docstring)."""

    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    is_greedy: bool


class LoadedModel(Protocol):
    """The call surface every backend (real or mock) must expose — the rest
    of the pipeline (generation/sampler.py, scoring/logprob.py) is written
    against this, never against a specific backend.

    Deliberately does NOT take a `contaminated` flag: a real quantized model
    must never receive the contamination ground-truth label as a generation
    input — that label is the withheld variable this whole design measures
    via output statistics (detector scores, pass rate), not something the
    model conditions on. `models/mock.py`'s synthetic generative process
    needs that ground truth to produce known-signed test outputs, but it
    gets it via `MockModel.register_item()` ahead of time, keyed by
    `item_id` — not through this shared call surface."""

    tokenizer: object

    def generate(self, item_id: str, prompt: str, *, temperature: float, sample_id: int): ...

    def score_logprobs(self, item_id: str, token_ids: list[int]) -> list[float]: ...


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
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: PLC0415

    if quant is Quant.BNB_INT8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant is Quant.BNB_NF4:
        # bnb_4bit_compute_dtype defaults to fp32 if unset — needlessly slow
        # on H100/Hopper's bf16 tensor cores; found while wiring up the real
        # loading path, not a paper-relevant numerical choice (the stored
        # weights are still nf4; this only affects the dequantized matmul
        # compute dtype).
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
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
    on mock-vs-real. Covered by scripts/run_smoke_test.py (real GPU) and
    tests/test_real_model_adapter.py (tiny CPU model, no chat template)."""

    def __init__(self, model, tokenizer, *, max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        # score_logprobs() only receives token_ids (matches the shared
        # LoadedModel Protocol, which has no prompt argument) but a real
        # teacher-forced pass needs the prompt as context. generate() is
        # always called before score_logprobs() for a given item at every
        # current call site (generation/sampler.py), so stash it here on
        # first generate() and look it up when needed.
        self._prompts: dict[str, str] = {}

    def _build_input_ids(self, prompt: str):
        # Returns (input_ids, attention_mask). apply_chat_template(...,
        # return_tensors="pt") returns a BatchEncoding (dict-like with
        # input_ids/attention_mask), not a bare tensor, on the installed
        # transformers version — found by the smoke test: passing the
        # BatchEncoding itself as model.generate()'s positional `inputs`
        # fails inside generate() with an opaque AttributeError
        # (BatchEncoding.__getattr__ has no .shape). Always unpack both
        # tensors explicitly and pass attention_mask through to
        # generate()/the forward pass rather than relying on padding-free
        # single-sequence generation to make it optional.
        if getattr(self.tokenizer, "chat_template", None):
            encoded = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt")
        return encoded["input_ids"].to(self.model.device), encoded["attention_mask"].to(self.model.device)

    def generate(self, item_id: str, prompt: str, *, temperature: float, sample_id: int) -> _RealGenerationSample:
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        self._prompts[item_id] = prompt
        input_ids, attention_mask = self._build_input_ids(prompt)
        is_greedy = temperature == 0.0

        torch.manual_seed(_seed_from(item_id, sample_id, temperature))

        gen_kwargs: dict = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=not is_greedy,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if not is_greedy:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

        prompt_len = input_ids.shape[-1]
        new_token_ids = outputs.sequences[0][prompt_len:].tolist()

        # outputs.scores[i] is the pre-softmax logits for generation step i
        # (one entry per new token, in order) — reuse them instead of a
        # second forward pass to get each chosen token's log-probability.
        token_logprobs = [
            F.log_softmax(step_logits[0].float(), dim=-1)[token_id].item()
            for step_logits, token_id in zip(outputs.scores, new_token_ids)
        ]

        text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return _RealGenerationSample(text=text, token_ids=new_token_ids, token_logprobs=token_logprobs, is_greedy=is_greedy)

    def score_logprobs(self, item_id: str, token_ids: list[int]) -> list[float]:
        """Teacher-forced per-token log-probability for an already-generated
        completion (token_ids), conditioned on the prompt this item_id was
        last generate()-d with."""
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        try:
            prompt = self._prompts[item_id]
        except KeyError:
            raise RuntimeError(
                f"score_logprobs(item_id={item_id!r}) called before generate() for this item on this "
                "adapter instance — a real backend needs the prompt as context and only tracks it via "
                "generate()'s call history (the shared LoadedModel Protocol carries no prompt argument)."
            ) from None

        prompt_ids, prompt_attention_mask = self._build_input_ids(prompt)
        completion_ids = torch.tensor([token_ids], dtype=torch.long, device=prompt_ids.device)
        completion_attention_mask = torch.ones_like(completion_ids)
        full_ids = torch.cat([prompt_ids, completion_ids], dim=-1)
        full_attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=-1)

        with torch.no_grad():
            logits = self.model(full_ids, attention_mask=full_attention_mask).logits

        prompt_len = prompt_ids.shape[-1]
        # logits[:, prompt_len - 1] predicts the first completion token, ...,
        # logits[:, -2] predicts the last one.
        completion_logits = logits[0, prompt_len - 1 : -1, :].float()
        log_probs = F.log_softmax(completion_logits, dim=-1)
        return [log_probs[i, token_id].item() for i, token_id in enumerate(token_ids)]
