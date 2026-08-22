"""load_model(spec, quant) — branches bf16/bnb-int8/bnb-nf4/gptq-awq-int4/mock.

All real (non-mock) backends lazy-import their heavy dependencies inside the
branch functions, not at module scope. This lets `import qcd.models.loader`
succeed on the mock-only local profile (no torch/transformers-with-torch/
bitsandbytes installed) — only actually calling a real backend requires the
H100 GPU stack (requirements-h100.txt).

**GPTQ_AWQ_INT4 is a historical enum value. The fourth rung is implemented
via AWQ only, through llm-compressor, uniformly for every model; GPTQ is not
implemented.** See `scripts/quantize_model.py`
for the offline quantization step this backend loads from (quantize once
and save, unlike bnb's load-time quantization).
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Protocol

from qcd.config import ModelSpec, Quant
from qcd.models.mock import MockModel, MockTokenizer

# Engineering default, not a paper-derived constant (constants.py is scoped to
# AGENTS.md/paper_draft.md-derived statistical/design values) — deliberately
# separate from that module.
_DEFAULT_MAX_NEW_TOKENS = 512

# Repo-root-anchored regardless of invoking CWD (matches
# scripts/run_smoke_test.py's/scripts/quantize_model.py's own convention) —
# lands under the gitignored `/data/` directory, which is root-anchored in
# .gitignore, not `pipeline/data/`.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_QUANTIZED_DIR = _REPO_ROOT / "data" / "quantized"


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

    def score_prompt_logprobs(self, item_id: str, prompt: str) -> list[float]: ...


def load_model(spec: ModelSpec, quant: Quant, *, mock: bool = False) -> LoadedModel:
    if mock:
        return MockModel(MockTokenizer())

    backend = {
        Quant.BF16: _load_bf16,
        Quant.BNB_INT8: _load_bnb,
        Quant.BNB_NF4: _load_bnb,
        Quant.GPTQ_AWQ_INT4: _load_gptq_or_awq,
    }[quant]
    return backend(spec, quant)


def _load_bf16(spec: ModelSpec, quant: Quant) -> LoadedModel:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo_id, revision=spec.revision)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_repo_id, revision=spec.revision, dtype=torch.bfloat16, device_map="auto"
    )
    return _RealModelAdapter(model, tokenizer, revision=spec.revision)


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

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo_id, revision=spec.revision)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_repo_id, revision=spec.revision,
        quantization_config=bnb_config, device_map="auto",
    )
    return _RealModelAdapter(model, tokenizer, revision=spec.revision)


def _quantized_checkpoint_dir(spec: ModelSpec) -> Path:
    """The one **canonical** local AWQ checkpoint path for a model — no
    calibration-domain suffix. `scripts/quantize_model.py` saves to a
    calibration-tagged directory (`<model>-awq-code`/`<model>-awq-chat`) for
    comparison purposes; this function does not resolve those — the winning
    variant must be copied/re-quantized to this exact path by hand, a
    deliberate step (see module docstring)."""
    return _QUANTIZED_DIR / f"{spec.name}-awq"


def _load_gptq_or_awq(spec: ModelSpec, quant: Quant) -> LoadedModel:
    checkpoint_dir = _quantized_checkpoint_dir(spec)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"no quantized AWQ checkpoint at {checkpoint_dir} for {spec.name!r} — quantization is "
            "a deliberate, separate offline step, never implicit inside a real run. Run "
            f"`python scripts/quantize_model.py {spec.name} --calibration <code|chat>`, then copy or "
            "re-quantize the chosen calibration variant to this exact path (see module docstring)."
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, device_map="auto")
    manifest = checkpoint_dir / "quantization_manifest.json"
    revision = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None
    if revision is None:
        raise FileNotFoundError(f"AWQ checkpoint is missing {manifest}")
    return _RealModelAdapter(model, tokenizer, revision=revision)


class _RealModelAdapter:
    """Wraps a real transformers model+tokenizer to expose the same
    generate()/score_logprobs() surface as MockModel, so callers never branch
    on mock-vs-real. Covered by scripts/run_smoke_test.py (real GPU) and
    tests/test_real_model_adapter.py (tiny CPU model, no chat template)."""

    def __init__(
        self, model, tokenizer, *, max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        revision: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.revision = revision
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

    def score_prompt_logprobs(self, item_id: str, prompt: str) -> list[float]:
        """Teacher-force only benchmark-prompt tokens in model context.

        Instruct models still receive their normal chat wrapper and assistant
        generation marker.  Offset mappings select only tokens wholly inside
        the literal user prompt, excluding wrapper/special tokens from the
        returned statistic.  Unlike ``score_logprobs``, this API is independent
        of generation history and is therefore safe when generations came from
        a persistent cache.
        """
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415

        del item_id  # kept in the shared API for item-level tracing symmetry
        has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        if has_chat_template:
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            rendered = prompt

        prompt_start = rendered.find(prompt)
        if prompt_start < 0:
            raise ValueError("chat template did not preserve the benchmark prompt verbatim")
        prompt_end = prompt_start + len(prompt)

        try:
            encoded = self.tokenizer(
                rendered,
                add_special_tokens=not has_chat_template,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            target_positions = [
                position
                for position, (token_start, token_end) in enumerate(offsets)
                if token_end > token_start
                and token_start >= prompt_start
                and token_end <= prompt_end
            ]
        except (NotImplementedError, TypeError):
            # Slow tokenizers may not expose offsets. This fallback uses the
            # fully-contained prefix span; target-model tokenizers are fast
            # tokenizers and take the exact offset path above.
            prefix_ids = self.tokenizer(
                rendered[:prompt_start], add_special_tokens=False
            )["input_ids"]
            through_prompt_ids = self.tokenizer(
                rendered[:prompt_end], add_special_tokens=False
            )["input_ids"]
            encoded = self.tokenizer(
                rendered, add_special_tokens=False, return_tensors="pt"
            )
            target_positions = list(range(len(prefix_ids), len(through_prompt_ids)))

        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)
        # Position zero has no causal left context unless a BOS token precedes
        # it, so it cannot supply a next-token probability.
        target_positions = [position for position in target_positions if position > 0]
        if not target_positions:
            raise ValueError("benchmark prompt has no token with causal left context")

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits[0]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return [
            log_probs[position - 1, input_ids[0, position]].item()
            for position in target_positions
        ]
