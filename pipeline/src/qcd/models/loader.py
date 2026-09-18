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
and save, unlike bnb's load-time quantization). That step also writes paper
§4.3's calibration/evaluation-prompt overlap report, without which this
module refuses to load an AWQ checkpoint.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Protocol

from qcd.config import ModelSpec, Quant
from qcd.constants import (
    DECODING_GREEDY_TEMPERATURE, DECODING_LENGTH_PENALTY,
    DECODING_MIN_NEW_TOKENS, DECODING_REPETITION_PENALTY,
    DECODING_TOP_K_DISABLED, DECODING_TOP_P, FOLLOW_CHECKPOINT_GENERATION_CONFIG,
    GENERATION_MAX_NEW_TOKENS,
)
from qcd.data.schema import PromptScoringDetail
from qcd.io.manifest import require_calibration_overlap_report
from qcd.models.mock import MockModel, MockTokenizer

# Paper §4.4's "512-token generation cap" — this used to be a local
# engineering default here; it is now pinned in constants.py because §4.4
# fixes it and real_run.py has to record the same number in the manifest.
_DEFAULT_MAX_NEW_TOKENS = GENERATION_MAX_NEW_TOKENS

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
    docstring).

    `token_logprobs` holds **raw** per-token log-probabilities (see
    `_RealModelAdapter.generate`), and `truncated_at_cap` records whether this
    generation stopped because it hit the 512-token cap rather than because
    the model emitted a stop token (paper §4.4: "we record per item whether
    generation stopped at the cap and report the truncated-generation rate by
    precision")."""

    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    is_greedy: bool
    truncated_at_cap: bool = False


# --- Frozen decoding settings (paper §4.4) ---------------------------------
#
# §4.4: "Every decoding setting is specified explicitly and the checkpoint's
# own `generation_config` is not followed."
#
# How transformers 5.14.1 resolves a generation config (generation/utils.py
# `_prepare_generation_config`, lines 1733-1773), verified by reading the
# installed source:
#   1. the `generation_config` argument (or a bare `GenerationConfig()` when
#      none is passed) is deep-copied;
#   2. `generation_config.update(**self.generation_config.to_dict(),
#      defaults_only=True, ...)` (line 1769) fills in every field that is
#      still `None` from the **checkpoint's** generation_config.json;
#   3. `generation_config.update(**global_defaults, defaults_only=True)`
#      (line 1770) fills whatever is still `None` from the library defaults
#      (`GenerationConfig._get_default_generation_params()`,
#      configuration_utils.py:590).
# `update(..., defaults_only=True)` only overwrites fields whose current value
# is `None` (configuration_utils.py:1326), and every sampling field defaults
# to `None` in `GenerationConfig.__init__` (configuration_utils.py:390-442).
# So a field we set explicitly survives step 2 unchanged, and a field we leave
# unset inherits the checkpoint's value.
#
# That gives two rules, both used below:
#   - pin every decoding knob whose neutral value is expressible (top_p=1.0,
#     top_k=0, repetition_penalty=1.0, ...);
#   - for knobs whose only "off" value is `None` (bad_words_ids,
#     suppress_tokens, penalty_alpha, sequence_bias, ...) pinning is
#     impossible — `None` means "unset" — so the adapter additionally replaces
#     the loaded model's own `generation_config` with a neutral one that
#     carries nothing but the checkpoint's stop/pad token ids
#     (`_RealModelAdapter._neutralize_checkpoint_generation_config`).

# Decoding fields recorded in the run manifest. Deliberately excludes token
# ids and cache/compile settings: those are model identity and performance,
# not decoding policy. `max_length` is excluded too — transformers overwrites
# it per prompt with `input_length + max_new_tokens` (utils.py:1696), so the
# only stable number is `max_new_tokens`.
_RECORDED_DECODING_KEYS: tuple[str, ...] = (
    "max_new_tokens", "min_length", "min_new_tokens", "max_time",
    "stop_strings", "early_stopping", "do_sample", "num_beams",
    "num_beam_groups", "num_return_sequences", "diversity_penalty",
    "penalty_alpha", "dola_layers", "temperature", "top_k", "top_p", "top_h",
    "min_p", "typical_p", "epsilon_cutoff", "eta_cutoff",
    "repetition_penalty", "encoder_repetition_penalty", "no_repeat_ngram_size",
    "encoder_no_repeat_ngram_size", "bad_words_ids", "sequence_bias",
    "guidance_scale", "length_penalty", "forced_bos_token_id",
    "forced_eos_token_id", "exponential_decay_length_penalty",
    "suppress_tokens", "begin_suppress_tokens", "remove_invalid_values",
    "renormalize_logits", "watermarking_config", "constraints",
    "force_words_ids", "use_mtp", "prompt_lookup_num_tokens",
    "assistant_early_exit", "use_cache", "output_logits", "output_scores",
    "return_dict_in_generate",
)


def frozen_decoding_settings(
    *, temperature: float, max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS
) -> dict:
    """Every decoding knob we pin explicitly, as `GenerationConfig` kwargs.

    `temperature == 0.0` means the greedy reference output: §4.4 says it
    "differs from the samples only in that sampling is off", so only
    `do_sample` changes, and the recorded temperature becomes the no-op 1.0.

    Fields left out of this dict are the ones whose neutral value is `None`
    and therefore cannot be pinned (see the module-level note above); they are
    handled by neutralizing the model's own generation_config instead.
    """
    is_greedy = temperature == 0.0
    return {
        # Length: §4.4's 512-token cap, and "no ... minimum-length constraint".
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": DECODING_MIN_NEW_TOKENS,
        "min_length": DECODING_MIN_NEW_TOKENS,
        # §4.4: "no length penalty". length_penalty only bites in beam search,
        # but it is pinned so the recorded settings are complete.
        "length_penalty": DECODING_LENGTH_PENALTY,
        "early_stopping": False,
        "num_beams": 1,
        "num_beam_groups": 1,
        "diversity_penalty": 0.0,
        "num_return_sequences": 1,
        # Sampling: temperature only, per §4.4.
        "do_sample": not is_greedy,
        "temperature": DECODING_GREEDY_TEMPERATURE if is_greedy else temperature,
        "top_p": DECODING_TOP_P,
        "top_k": DECODING_TOP_K_DISABLED,
        "typical_p": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        # §4.4: "repetition_penalty=1.0". This one also applies to the greedy
        # output, which is why it is pinned on both paths.
        "repetition_penalty": DECODING_REPETITION_PENALTY,
        "encoder_repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "encoder_no_repeat_ngram_size": 0,
        "remove_invalid_values": False,
        "renormalize_logits": False,
        "use_mtp": False,
        "use_cache": True,
        # Raw (pre-logits-processor) logits, needed for the stored token
        # log-probabilities — see `_RealModelAdapter.generate`.
        "output_logits": True,
        "output_scores": False,
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict_in_generate": True,
    }


def build_generation_config(
    *,
    temperature: float,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    pad_token_id: int | None = None,
):
    """The `GenerationConfig` object handed to `model.generate()`.

    Passing a config object (rather than loose `generate(**kwargs)`) is what
    makes the checkpoint's own values lose: they are only applied to fields
    still `None` at step 2 of the resolution order documented above.
    """
    from transformers import GenerationConfig  # noqa: PLC0415

    settings = frozen_decoding_settings(
        temperature=temperature, max_new_tokens=max_new_tokens
    )
    if pad_token_id is not None:
        settings["pad_token_id"] = pad_token_id
    return GenerationConfig(**settings)


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def resolved_decoding_settings(
    *, temperature: float, max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS
) -> dict:
    """What `generate()` actually runs with, for the run manifest.

    Reproduces transformers' own resolution (`_prepare_generation_config`
    step 3) by applying `GenerationConfig._get_default_generation_params()` to
    the fields we left unset. Step 2 — the checkpoint's generation_config — is
    skipped on purpose: the adapter neutralizes it, so it contributes nothing,
    which is what makes this record model-independent and computable before
    any checkpoint is loaded.
    """
    from transformers import GenerationConfig  # noqa: PLC0415

    explicit = frozen_decoding_settings(
        temperature=temperature, max_new_tokens=max_new_tokens
    )
    resolved = copy.deepcopy(build_generation_config(
        temperature=temperature, max_new_tokens=max_new_tokens
    ))
    resolved.update(
        **GenerationConfig._get_default_generation_params(), defaults_only=True
    )
    return {
        "follows_checkpoint_generation_config": FOLLOW_CHECKPOINT_GENERATION_CONFIG,
        "requested_temperature": temperature,
        "explicitly_set": sorted(explicit),
        "resolved": {
            key: _jsonable(getattr(resolved, key, None))
            for key in _RECORDED_DECODING_KEYS
        },
    }


def decoding_settings_id(settings: dict) -> str:
    """Short content address of one resolved decoding-settings record, stored
    on every generations row so a row can be tied back to the manifest's full
    record (and so a settings change misses the generation cache)."""
    canonical = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def hit_length_cap(
    token_ids: list[int], max_new_tokens: int, eos_token_ids: frozenset[int]
) -> bool:
    """True when a generation stopped at the token cap rather than on a stop
    token (paper §4.4's per-item truncation record).

    A generation that emits EOS exactly at the cap is *not* truncated: the
    stop token is present, so the model finished. Below the cap, generation
    can only have stopped on a stop token."""
    if len(token_ids) < max_new_tokens:
        return False
    if token_ids and token_ids[-1] in eos_token_ids:
        return False
    return True


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

    # `_RealModelAdapter` additionally offers `score_prompt_detail()`, which
    # returns the same log-probabilities plus paper §4.4's provenance fields
    # (target text, token boundaries, chat template). It is deliberately NOT
    # part of this Protocol: `models/mock.py` has no chat template or token
    # offsets to report, so `real_run.py` treats it as optional and falls back
    # to `score_prompt_logprobs()` when a backend does not provide it.


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
    """The one **canonical** local AWQ checkpoint path for a model.

    Paper §4.3: "Exactly one calibration artifact per model enters the main
    run." `scripts/quantize_model.py` writes directly here, with the frozen
    code calibration, for every model — there is no variant-comparison step
    and no hand copy."""
    return _QUANTIZED_DIR / f"{spec.name}-awq"


def _load_gptq_or_awq(spec: ModelSpec, quant: Quant) -> LoadedModel:
    checkpoint_dir = _quantized_checkpoint_dir(spec)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"no quantized AWQ checkpoint at {checkpoint_dir} for {spec.name!r} — quantization is "
            "a deliberate, separate offline step, never implicit inside a real run. Run "
            f"`python scripts/quantize_model.py {spec.name}`, which writes this exact path."
        )
    # Paper §4.3's overlap check is a *pre-execution* requirement, so a
    # checkpoint whose calibration text was never checked against the
    # evaluation prompts cannot be loaded at all. Checked before the heavy
    # imports, like the directory check above.
    require_calibration_overlap_report(checkpoint_dir)

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
        # Stop tokens are model identity, not a decoding setting, so they are
        # read from the checkpoint before its generation_config is replaced.
        self.eos_token_ids = _collect_eos_token_ids(model, tokenizer)
        self.checkpoint_generation_config = getattr(model, "generation_config", None)
        self._neutralize_checkpoint_generation_config()
        self._generation_configs: dict[float, object] = {}
        # score_logprobs() only receives token_ids (matches the shared
        # LoadedModel Protocol, which has no prompt argument) but a real
        # teacher-forced pass needs the prompt as context. generate() is
        # always called before score_logprobs() for a given item at every
        # current call site (generation/sampler.py), so stash it here on
        # first generate() and look it up when needed.
        self._prompts: dict[str, str] = {}

    def _neutralize_checkpoint_generation_config(self) -> None:
        """Replace the checkpoint's `generation_config` with one that carries
        only its stop/pad/start token ids.

        Paper §4.4: "the checkpoint's own `generation_config` is not
        followed". Passing our own config to `generate()` already wins for
        every field we can pin, but a knob whose neutral value is `None`
        (`bad_words_ids`, `suppress_tokens`, `sequence_bias`, `penalty_alpha`,
        `min_p`, `forced_eos_token_id`, ...) cannot be pinned, because
        transformers reads `None` as "unset" and refills it from the
        checkpoint (generation/utils.py:1769). Emptying the checkpoint's
        config closes that path: after this, steps 2 and 3 of the resolution
        order can only contribute the library's own documented defaults.

        Token ids are deliberately carried over — dropping `eos_token_id`
        would stop generation from ever terminating before the 512-token cap
        and would silently inflate the truncation rate §4.4 asks us to
        report.
        """
        original = getattr(self.model, "generation_config", None)
        if original is None:
            return

        from transformers import GenerationConfig  # noqa: PLC0415

        carried = {
            name: getattr(original, name, None)
            for name in (
                "bos_token_id", "eos_token_id", "pad_token_id",
                "decoder_start_token_id",
            )
        }
        self.model.generation_config = GenerationConfig(
            **{name: value for name, value in carried.items() if value is not None}
        )

    def _generation_config_for(self, temperature: float):
        """One frozen config per temperature, built once and reused —
        `generate()` deep-copies it, so reuse is safe."""
        config = self._generation_configs.get(temperature)
        if config is None:
            pad_token_id = (
                getattr(self.tokenizer, "pad_token_id", None)
                or getattr(self.tokenizer, "eos_token_id", None)
            )
            config = build_generation_config(
                temperature=temperature,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=pad_token_id,
            )
            self._generation_configs[temperature] = config
        return config

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

        # Every decoding setting travels in this object, not as loose
        # generate() kwargs — that is what stops the checkpoint's own
        # generation_config from filling them in (paper §4.4; see the
        # module-level resolution-order note). `attention_mask` stays a kwarg
        # because it is a model input, not a decoding setting.
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=self._generation_config_for(temperature),
            )

        prompt_len = input_ids.shape[-1]
        new_token_ids = outputs.sequences[0][prompt_len:].tolist()

        # outputs.logits[i] is the **raw**, unprocessed logits for generation
        # step i (one entry per new token, in order), captured before the
        # logits processors run (transformers 5.14.1 generation/utils.py:2907
        # and :2917); outputs.scores would be the post-processor values.
        # §4.4's probability quantities are raw model log-probabilities, so
        # this path uses the raw logits. With the frozen settings above the
        # two agree for greedy decoding and differ only by the temperature
        # warper for samples, but the stored number is the raw one either way.
        #
        # These values are stored as `token_logprobs` in generations.parquet
        # and feed real_run.py's `completion_perplexity` /
        # `completion_mink_prob` diagnostic scores. The Q1 perplexity and
        # Min-k% scores do NOT come from here — they come from
        # score_prompt_detail()'s fixed-benchmark-text pass.
        token_logprobs = [
            F.log_softmax(step_logits[0].float(), dim=-1)[token_id].item()
            for step_logits, token_id in zip(outputs.logits, new_token_ids)
        ]

        text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return _RealGenerationSample(
            text=text, token_ids=new_token_ids, token_logprobs=token_logprobs,
            is_greedy=is_greedy,
            truncated_at_cap=hit_length_cap(
                new_token_ids, self.max_new_tokens, self.eos_token_ids
            ),
        )

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

        Unchanged signature (it is the shared ``LoadedModel`` surface, which
        ``models/mock.py`` also implements); ``score_prompt_detail`` returns
        the same numbers plus the §4.4 provenance fields.
        """
        return self.score_prompt_detail(item_id, prompt).logprobs

    def score_prompt_detail(self, item_id: str, prompt: str) -> PromptScoringDetail:
        """``score_prompt_logprobs`` plus what paper §4.4 asks to be recorded
        with a probability-detector score: the scored (target) text, the token
        boundaries that were scored, and the chat template that rendered
        them."""
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
        logprobs = [
            log_probs[position - 1, input_ids[0, position]].item()
            for position in target_positions
        ]
        return PromptScoringDetail(
            logprobs=logprobs,
            target_token_indices=list(target_positions),
            target_char_span=(prompt_start, prompt_end),
            rendered_char_length=len(rendered),
            chat_template_applied=has_chat_template,
            chat_template=getattr(self.tokenizer, "chat_template", None),
            target_text=prompt,
        )


def _collect_eos_token_ids(model, tokenizer) -> frozenset[int]:
    """Every id that ends a generation, from the checkpoint's own
    generation_config (which may carry several) and the tokenizer.

    Read at adapter construction, before the checkpoint's generation_config is
    neutralized. Only used to tell "stopped on a stop token" from "stopped at
    the 512-token cap" (paper §4.4's per-item truncation record)."""
    ids: set[int] = set()
    sources = [
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ]
    for source in sources:
        if source is None:
            continue
        if isinstance(source, int):
            ids.add(source)
        else:
            ids.update(int(value) for value in source)
    return frozenset(ids)
