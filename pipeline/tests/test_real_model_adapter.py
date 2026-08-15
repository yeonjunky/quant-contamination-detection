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
