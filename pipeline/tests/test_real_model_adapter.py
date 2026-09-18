"""CPU-only regression coverage for _RealModelAdapter (models/loader.py's real
generate()/score_logprobs() implementation), independent of the GPU-scale nf4
smoke test (scripts/run_smoke_test.py needs a real 7B download + GPU every
run). Uses a tiny public checkpoint with no chat_template, so it also
exercises the plain-tokenization fallback path (the real target models —
Qwen2.5/Llama-3.1/Olmo3, all -Instruct — take the chat-template branch
instead, only covered by the real smoke test).

Gated with pytest.importorskip so the mock-only/no-torch profile is
unaffected — this file is simply skipped when torch isn't installed.
"""

import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from qcd.models.loader import _RealModelAdapter  # noqa: E402

_TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def adapter() -> _RealModelAdapter:
    tokenizer = AutoTokenizer.from_pretrained(_TINY_MODEL)
    model = AutoModelForCausalLM.from_pretrained(_TINY_MODEL)
    # Small max_new_tokens keeps this test fast; unrelated to the real
    # smoke test's _DEFAULT_MAX_NEW_TOKENS.
    return _RealModelAdapter(model, tokenizer, max_new_tokens=8)


def test_no_chat_template_on_tiny_model():
    # Confirms this test actually exercises the plain-tokenization fallback
    # branch in _RealModelAdapter._build_input_ids, not the chat-template one.
    tokenizer = AutoTokenizer.from_pretrained(_TINY_MODEL)
    assert tokenizer.chat_template is None


def test_generate_returns_matching_length_token_ids_and_logprobs(adapter):
    sample = adapter.generate("item-1", "def add(a, b):\n    return", temperature=0.0, sample_id=0)
    assert len(sample.token_ids) == len(sample.token_logprobs)
    assert len(sample.token_ids) > 0
    assert len(sample.token_ids) <= 8
    assert sample.is_greedy is True
    assert isinstance(sample.text, str)


def test_greedy_generation_is_deterministic(adapter):
    first = adapter.generate("item-2", "def add(a, b):\n    return", temperature=0.0, sample_id=0)
    second = adapter.generate("item-2", "def add(a, b):\n    return", temperature=0.0, sample_id=0)
    assert first.token_ids == second.token_ids


def test_score_logprobs_returns_finite_values_of_requested_length(adapter):
    sample = adapter.generate("item-3", "def add(a, b):\n    return", temperature=0.0, sample_id=0)
    scores = adapter.score_logprobs("item-3", sample.token_ids)
    assert len(scores) == len(sample.token_ids)
    assert all(torch.isfinite(torch.tensor(scores)))


def test_score_logprobs_before_generate_raises_runtime_error(adapter):
    with pytest.raises(RuntimeError, match="called before generate"):
        adapter.score_logprobs("never-generated", [1, 2, 3])


def test_score_prompt_logprobs_is_generation_independent(adapter):
    scores = adapter.score_prompt_logprobs(
        "never-generated", "def add(a, b):\n    return a + b"
    )
    assert scores
    assert all(torch.isfinite(torch.tensor(scores)))


def test_score_prompt_logprobs_excludes_chat_wrapper_tokens():
    class CharacterChatTokenizer:
        chat_template = "test-template"

        def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
            assert add_generation_prompt is True
            assert tokenize is False
            return "<user>" + messages[0]["content"] + "</user><assistant>"

        def __call__(
            self, text, *, add_special_tokens=False,
            return_offsets_mapping=False, return_tensors=None,
        ):
            del add_special_tokens
            ids = [ord(char) for char in text]
            result = {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
            if return_offsets_mapping:
                result["offset_mapping"] = torch.tensor(
                    [[(i, i + 1) for i in range(len(text))]]
                )
            assert return_tensors in (None, "pt")
            return result

    class UniformModel:
        device = torch.device("cpu")

        def __call__(self, input_ids, *, attention_mask):
            del attention_mask
            logits = torch.zeros((1, input_ids.shape[1], 128))
            return SimpleNamespace(logits=logits)

    prompt = "XY"
    chat_adapter = _RealModelAdapter(UniformModel(), CharacterChatTokenizer())
    scores = chat_adapter.score_prompt_logprobs("item", prompt)
    assert len(scores) == len(prompt)
    assert scores == pytest.approx([-torch.log(torch.tensor(128.0)).item()] * 2)


# --- paper §4.4's frozen decoding settings, end to end ----------------------
#
# The torch-free counterparts live in tests/test_decoding_settings.py; these
# are the ones that actually call transformers' `generate()`, so they only run
# where torch is installed (the H100 profile), not on the mock-only profile.


def _tiny_model_with_checkpoint_decoding_settings():
    """A checkpoint that ships the settings paper §4.4 warns about — the
    Qwen2.5-32B-Instruct example: temperature 0.7, top_p 0.8, top_k 20,
    repetition_penalty 1.05."""
    model = AutoModelForCausalLM.from_pretrained(_TINY_MODEL)
    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.7
    model.generation_config.top_p = 0.8
    model.generation_config.top_k = 20
    model.generation_config.repetition_penalty = 1.05
    return model


def test_checkpoint_generation_config_does_not_reach_generate():
    tokenizer = AutoTokenizer.from_pretrained(_TINY_MODEL)
    prompt = "def add(a, b):\n    return"

    loaded = _RealModelAdapter(
        _tiny_model_with_checkpoint_decoding_settings(), tokenizer, max_new_tokens=16
    )
    clean = _RealModelAdapter(
        AutoModelForCausalLM.from_pretrained(_TINY_MODEL), tokenizer, max_new_tokens=16
    )

    from_loaded = loaded.generate("item", prompt, temperature=0.0, sample_id=0)
    from_clean = clean.generate("item", prompt, temperature=0.0, sample_id=0)

    # Identical greedy output and identical log-probabilities: the shipped
    # repetition penalty (which applies to greedy decoding too) and the shipped
    # top-k/top-p had no effect.
    assert from_loaded.token_ids == from_clean.token_ids
    assert from_loaded.token_logprobs == pytest.approx(from_clean.token_logprobs)


def test_sampling_uses_the_frozen_settings_not_the_checkpoints():
    tokenizer = AutoTokenizer.from_pretrained(_TINY_MODEL)
    adapter = _RealModelAdapter(
        _tiny_model_with_checkpoint_decoding_settings(), tokenizer, max_new_tokens=16
    )

    config = adapter._generation_config_for(0.8)
    assert config.temperature == 0.8       # not the checkpoint's 0.7
    assert config.top_p == 1.0             # not 0.8
    assert config.top_k == 0               # not 20
    assert config.repetition_penalty == 1.0  # not 1.05
    assert adapter.model.generation_config.top_k is None


def test_generate_token_logprobs_are_raw_teacher_forced_values(adapter):
    sample = adapter.generate("raw-logprob", "def add(a, b):\n    return", temperature=0.0, sample_id=0)

    # score_logprobs() is an independent teacher-forced forward pass over the
    # same tokens with no logits processors at all, so it is the reference for
    # "raw". outputs.scores would only match it while every processor is a
    # no-op — which is exactly what we must not depend on.
    teacher_forced = adapter.score_logprobs("raw-logprob", sample.token_ids)
    assert sample.token_logprobs == pytest.approx(teacher_forced, abs=1e-4)


def test_generation_that_runs_into_the_cap_is_flagged():
    tokenizer = AutoTokenizer.from_pretrained(_TINY_MODEL)
    model = AutoModelForCausalLM.from_pretrained(_TINY_MODEL)
    adapter = _RealModelAdapter(model, tokenizer, max_new_tokens=4)

    sample = adapter.generate("truncation", "def add(a, b):\n    return", temperature=0.0, sample_id=0)

    if len(sample.token_ids) == 4 and sample.token_ids[-1] not in adapter.eos_token_ids:
        assert sample.truncated_at_cap is True
    else:  # the random tiny model happened to emit a stop token first
        assert sample.truncated_at_cap is False


def test_score_prompt_detail_reports_boundaries_and_template(adapter):
    prompt = "def add(a, b):\n    return a + b"
    detail = adapter.score_prompt_detail("detail", prompt)

    assert detail.logprobs == pytest.approx(adapter.score_prompt_logprobs("detail", prompt))
    assert len(detail.target_token_indices) == len(detail.logprobs)
    start, end = detail.target_char_span
    assert end - start == len(prompt)
    # The tiny checkpoint has no chat template, so the plain-tokenization
    # fallback reports that honestly rather than inventing one.
    assert detail.chat_template_applied is False
    assert detail.chat_template is None
    assert detail.target_text == prompt
