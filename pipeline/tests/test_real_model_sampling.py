"""`_RealModelAdapter.generate_samples` against a real `model.generate()`:
a tiny randomly initialized Llama built from a config, so nothing is
downloaded. Skipped when torch is not installed.

CPU fp32 only. Whether the same holds for bf16/quantized kernels on a GPU is
not covered here."""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from qcd.generation.cache import GenerationCache  # noqa: E402
from qcd.generation.sampler import sample_item  # noqa: E402
from qcd.models.loader import _RealModelAdapter, _seed_from  # noqa: E402

_EOS = 2
# Distinct from EOS so a padded (uncut) row would not end in a stop token and
# would read as truncated.
_PAD = 0
_MAX_NEW_TOKENS = 24
_PROMPT = "def add(a, b):"
_TEMPERATURE = 0.8


class _CharTokenizer:
    chat_template = None
    pad_token_id = _PAD
    eos_token_id = _EOS

    def __call__(self, text, return_tensors=None):
        del return_tensors
        ids = torch.tensor([[3 + ord(character) % 13 for character in text]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return " ".join(map(str, token_ids))


@pytest.fixture(scope="module")
def adapter() -> _RealModelAdapter:
    torch.manual_seed(0)
    # A 16-token vocabulary makes EOS likely within the cap, so some rows
    # stop early and others run to the cap. The large initializer_range makes
    # the next-token distributions peaked enough for the temperature to
    # change which token a given random draw selects.
    config = LlamaConfig(
        vocab_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        initializer_range=0.5,
        bos_token_id=1, eos_token_id=_EOS, pad_token_id=_PAD,
    )
    model = LlamaForCausalLM(config).eval()
    return _RealModelAdapter(model, _CharTokenizer(), max_new_tokens=_MAX_NEW_TOKENS)


def _samples(adapter, sample_ids, item_id="item"):
    return adapter.generate_samples(
        item_id, _PROMPT, temperature=_TEMPERATURE, sample_ids=list(sample_ids)
    )


def test_each_row_matches_its_sample_generated_alone(adapter):
    batch = _samples(adapter, range(8))
    for sample_id, row in enumerate(batch):
        [alone] = _samples(adapter, [sample_id])
        assert row.token_ids == alone.token_ids
        assert row.token_logprobs == pytest.approx(alone.token_logprobs, abs=1e-4)
        assert row.truncated_at_cap == alone.truncated_at_cap
    assert len({tuple(row.token_ids) for row in batch}) > 1


def test_the_same_batch_is_reproduced_exactly_whatever_the_global_rng(adapter):
    torch.manual_seed(1)
    first = _samples(adapter, range(8))
    torch.manual_seed(2)
    second = _samples(adapter, range(8))
    assert [row.token_ids for row in first] == [row.token_ids for row in second]
    assert [row.token_logprobs for row in first] == [row.token_logprobs for row in second]


def test_a_sample_id_keeps_its_tokens_when_the_row_order_changes(adapter):
    forward = _samples(adapter, range(8))
    backward = _samples(adapter, reversed(range(8)))
    assert [row.token_ids for row in forward] == [row.token_ids for row in reversed(backward)]


def test_chunked_sampling_gives_each_sample_id_the_same_tokens_as_one_batch(adapter, tmp_path):
    def sampled(batch_size):
        return sample_item(
            adapter, GenerationCache(tmp_path / f"cache{batch_size}"), model_name="tiny",
            quant="fp32", item_id="item", prompt=_PROMPT, n_samples=8,
            sample_temperature=_TEMPERATURE, batch_size=batch_size,
        ).samples

    chunked, whole = sampled(3), sampled(8)
    assert [row.token_ids for row in chunked] == [row.token_ids for row in whole]
    for chunk_row, whole_row in zip(chunked, whole):
        assert chunk_row.token_logprobs == pytest.approx(whole_row.token_logprobs, abs=1e-4)
    assert len({tuple(row.token_ids) for row in whole}) > 1


def test_seeds_depend_on_the_item(adapter):
    assert [row.token_ids for row in _samples(adapter, range(8), item_id="a")] != [
        row.token_ids for row in _samples(adapter, range(8), item_id="b")
    ]


def test_rows_that_stop_early_are_cut_right_after_their_stop_token(adapter):
    batch = _samples(adapter, range(8))
    stopped = [row for row in batch if _EOS in row.token_ids]
    ran_to_cap = [row for row in batch if _EOS not in row.token_ids]
    # Both kinds occur, and the batch ran on past the early stoppers.
    assert stopped and ran_to_cap
    assert any(len(row.token_ids) < _MAX_NEW_TOKENS for row in stopped)

    for row in stopped:
        assert row.token_ids.index(_EOS) == len(row.token_ids) - 1
        assert len(row.token_logprobs) == len(row.token_ids)
        assert row.truncated_at_cap is False
    for row in ran_to_cap:
        assert len(row.token_ids) == len(row.token_logprobs) == _MAX_NEW_TOKENS
        assert row.truncated_at_cap is True


def test_samples_use_the_frozen_sample_config_and_add_only_the_sampler(adapter, monkeypatch):
    received = {}
    original = adapter.model.generate

    def recording_generate(input_ids, **kwargs):
        received.update(kwargs, rows=input_ids.shape[0])
        return original(input_ids, **kwargs)

    monkeypatch.setattr(adapter.model, "generate", recording_generate)
    _samples(adapter, range(3))

    config = received["generation_config"]
    assert (config.do_sample, config.temperature, config.top_p, config.top_k) == (
        True, _TEMPERATURE, 1.0, 0,
    )
    # No per-step full-vocabulary logits are kept for the batch.
    assert config.output_logits is False
    assert received["rows"] == 3
    assert set(received) - {"rows"} == {
        "attention_mask", "generation_config", "logits_processor",
    }



def test_stored_logprobs_are_the_raw_model_logprobs(adapter):
    for row in _samples(adapter, range(4)):
        teacher_forced = adapter.score_logprobs("item", row.token_ids)
        assert row.token_logprobs == pytest.approx(teacher_forced, abs=1e-4)


def test_stored_logprobs_are_taken_before_the_temperature(adapter):
    input_ids, attention_mask = adapter._build_input_ids(_PROMPT)
    prompt_len = input_ids.shape[-1]
    for row in _samples(adapter, range(4)):
        completion = torch.tensor([row.token_ids])
        full_ids = torch.cat([input_ids, completion], dim=-1)
        with torch.no_grad():
            logits = adapter.model(full_ids, attention_mask=torch.ones_like(full_ids)).logits[0]
        step_logits = logits[prompt_len - 1 : -1]
        drawn = completion[0][:, None]
        raw = torch.log_softmax(step_logits, dim=-1).gather(1, drawn).squeeze(1).tolist()
        tempered = torch.log_softmax(step_logits / _TEMPERATURE, dim=-1).gather(1, drawn).squeeze(1).tolist()
        assert row.token_logprobs == pytest.approx(raw, abs=1e-4)
        assert max(abs(a - b) for a, b in zip(raw, tempered)) > 1e-2


def test_a_first_token_is_drawn_from_its_own_seeded_generator_at_the_temperature(adapter):
    """The seed policy itself: the token for (item_id, sample_id, temperature)
    is a multinomial draw from softmax(raw logits / temperature) with a
    generator seeded by `_seed_from(item_id, sample_id, temperature)`."""
    sample_ids = range(32)
    input_ids, attention_mask = adapter._build_input_ids(_PROMPT)
    with torch.no_grad():
        logits = adapter.model(input_ids, attention_mask=attention_mask).logits[0, -1]
    probs = torch.softmax(logits / _TEMPERATURE, dim=-1)
    expected = [
        torch.multinomial(
            probs, 1,
            generator=torch.Generator().manual_seed(_seed_from("item", s, _TEMPERATURE)),
        ).item()
        for s in sample_ids
    ]
    assert [row.token_ids[0] for row in _samples(adapter, sample_ids)] == expected


def _measure_script():
    path = Path(__file__).parents[1] / "scripts" / "measure_sample_batch.py"
    spec = importlib.util.spec_from_file_location("measure_sample_batch", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_main_run_path_adds_no_extra_logits_processor(adapter, monkeypatch):
    received = []
    original = adapter.model.generate

    def recording_generate(input_ids, **kwargs):
        received.append(kwargs.get("logits_processor"))
        return original(input_ids, **kwargs)

    monkeypatch.setattr(adapter.model, "generate", recording_generate)
    adapter.generate("item", _PROMPT, temperature=0.0, sample_id=0)
    _samples(adapter, range(3))

    assert adapter.extra_logits_processors == ()
    greedy, sampled = received
    assert greedy is None
    assert [type(processor).__name__ for processor in sampled] == ["_PerRowSampler"]


def test_forced_full_length_runs_every_row_to_the_cap_and_then_restores_decoding(adapter):
    measure = _measure_script()
    before = _samples(adapter, range(8))
    greedy_before = adapter.generate("item", _PROMPT, temperature=0.0, sample_id=0)
    assert any(len(row.token_ids) < _MAX_NEW_TOKENS for row in before)

    with measure.forced_full_length(adapter):
        forced = _samples(adapter, range(8))
        greedy_forced = adapter.generate("item", _PROMPT, temperature=0.0, sample_id=0)
    for row in [*forced, greedy_forced]:
        assert len(row.token_ids) == len(row.token_logprobs) == _MAX_NEW_TOKENS
        assert _EOS not in row.token_ids
        assert row.truncated_at_cap is True

    assert adapter.extra_logits_processors == ()
    after = _samples(adapter, range(8))
    assert [row.token_ids for row in after] == [row.token_ids for row in before]
    greedy_after = adapter.generate("item", _PROMPT, temperature=0.0, sample_id=0)
    assert greedy_after.token_ids == greedy_before.token_ids
