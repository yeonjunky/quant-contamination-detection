"""Paper §4.4's frozen decoding settings, the raw-log-probability path and
the truncation record — all without a GPU and without torch installed.

§4.4: "Every decoding setting is specified explicitly and the checkpoint's own
`generation_config` is not followed: `top_p=1.0`, top-k sampling disabled,
`repetition_penalty=1.0`, and no length penalty or minimum-length constraint.
The same settings apply to the greedy reference output, which differs from the
samples only in that sampling is off."

The checkpoint-override tests apply transformers' *own* merge steps (the two
`GenerationConfig.update(...)` calls at generation/utils.py:1769-1770 in
transformers 5.14.1, using the library's own `update`/`to_dict`/
`_get_default_generation_params`), because `generation/utils.py` itself cannot
be imported without torch. `tests/test_real_model_adapter.py` carries the
end-to-end counterparts that run `model.generate()` for real wherever torch is
installed.

`test_generate_*` drive the real `_RealModelAdapter.generate()` body against a
stub `torch` module, so they cover the parts that only a call can show: that
the stored log-probabilities come from `outputs.logits` (raw) and not
`outputs.scores` (post-logits-processor), and that the truncation flag is set
from the actual generation.
"""

import copy
import math
import sys

import pytest
from transformers import GenerationConfig

from qcd.constants import GENERATION_MAX_NEW_TOKENS
from qcd.models.loader import (
    _RealModelAdapter, build_generation_config, decoding_settings_id,
    frozen_decoding_settings, hit_length_cap, resolved_decoding_settings,
)


# Qwen2.5-32B-Instruct's shipped generation_config.json values, the example
# paper §4.4 names ("ships `temperature 0.7`, `top_p 0.8`, `top_k 20`,
# `repetition_penalty 1.05`"), plus two knobs whose only neutral value is
# `None` and which therefore cannot be pinned on our side.
def _checkpoint_generation_config() -> GenerationConfig:
    return GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        max_new_tokens=2048,
        eos_token_id=[151645, 151643],
        pad_token_id=151643,
        penalty_alpha=0.6,
        bad_words_ids=[[13]],
    )


def _merge_like_transformers(
    ours: GenerationConfig, checkpoint: GenerationConfig
) -> GenerationConfig:
    """transformers 5.14.1 `GenerationMixin._prepare_generation_config`,
    generation/utils.py:1767-1770, reproduced with the library's own API:
    deep-copy the caller's config, fill still-`None` fields from the
    checkpoint's config, then fill whatever is still `None` from the library
    defaults."""
    merged = copy.deepcopy(ours)
    merged.update(**checkpoint.to_dict(), defaults_only=True, allow_custom_entries=True)
    merged.update(
        **GenerationConfig._get_default_generation_params(), defaults_only=True
    )
    return merged


class _StubTokenizer:
    """No chat template, so `_build_input_ids` takes the plain-tokenization
    branch; ids are character codes so a fake generation is easy to read."""

    chat_template = None
    pad_token_id = None
    eos_token_id = 7

    def __call__(self, text, return_tensors=None):
        del return_tensors
        ids = [ord(character) for character in text]
        return {"input_ids": _FakeTensor([ids]), "attention_mask": _FakeTensor([[1] * len(ids)])}

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


class _FakeTensor:
    """The narrow slice of tensor behaviour `_RealModelAdapter.generate` uses:
    `.to(device)`, `.shape[-1]`, `[0]`, slicing and `.tolist()`."""

    def __init__(self, rows):
        self.rows = rows

    def to(self, device):
        del device
        return self

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def __getitem__(self, index):
        return _FakeRow(self.rows[index])


class _FakeRow:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _FakeRow(self.values[index])
        return self.values[index]

    def tolist(self):
        return list(self.values)

    def float(self):
        return _FakeRow(self.values)


class _FakeScalar(float):
    def item(self):
        return float(self)


class _FakeLogprobRow:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return _FakeScalar(self.values[index])


def _install_stub_torch(monkeypatch):
    """A `torch` just large enough for `_RealModelAdapter.generate`:
    `manual_seed`, `no_grad`, and `torch.nn.functional.log_softmax`."""
    import types

    torch_module = types.ModuleType("torch")
    torch_module.seeds = []

    def manual_seed(seed):
        torch_module.seeds.append(seed)

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *exc_info):
            return False

    torch_module.manual_seed = manual_seed
    torch_module.no_grad = _NoGrad

    functional = types.ModuleType("torch.nn.functional")

    def log_softmax(row, dim=-1):
        del dim
        values = row.values
        highest = max(values)
        total = sum(math.exp(value - highest) for value in values)
        return _FakeLogprobRow([value - highest - math.log(total) for value in values])

    functional.log_softmax = log_softmax
    nn_module = types.ModuleType("torch.nn")
    nn_module.functional = functional
    torch_module.nn = nn_module

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_module)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", functional)
    return torch_module


class _RecordingModel:
    """Returns a fixed generation, records the `generation_config` it was
    handed, and supplies raw `logits` that differ from processed `scores` — so
    a reader of the wrong field produces a visibly different number."""

    device = "cpu"

    def __init__(self, new_token_ids, *, generation_config=None):
        self.new_token_ids = new_token_ids
        self.generation_config = generation_config or _checkpoint_generation_config()
        self.received_config = None
        self.received_kwargs = None

    def generate(self, input_ids, **kwargs):
        self.received_config = kwargs.pop("generation_config", None)
        self.received_kwargs = kwargs
        prompt_ids = input_ids.rows[0]
        sequences = _FakeTensor([list(prompt_ids) + list(self.new_token_ids)])

        # Raw logits: the chosen token scores 0.0 and every other token -10.0.
        # "Processed" scores would instead have flattened that peak; nothing
        # in the adapter should ever see them.
        vocab_size = max(16, max(self.new_token_ids) + 1)
        raw_logits = []
        processed_scores = []
        for token_id in self.new_token_ids:
            raw = [-10.0] * vocab_size
            raw[token_id] = 0.0
            raw_logits.append(_FakeTensor2(raw))
            processed_scores.append(_FakeTensor2([0.0] * vocab_size))

        class _Output:
            pass

        output = _Output()
        output.sequences = sequences
        output.logits = tuple(raw_logits)
        output.scores = tuple(processed_scores)
        return output


class _FakeTensor2:
    """A one-row logits tensor: `step_logits[0].float()` yields the values."""

    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        assert index == 0
        return _FakeRow(self.values)


# --- (a) the checkpoint's generation_config is not followed ------------------


def test_frozen_settings_pin_the_values_paper_44_names():
    settings = frozen_decoding_settings(temperature=0.8)

    assert settings["top_p"] == 1.0
    assert settings["top_k"] == 0  # transformers disables top-k with 0, not None
    assert settings["repetition_penalty"] == 1.0
    assert settings["length_penalty"] == 1.0
    assert settings["min_length"] == 0
    assert settings["min_new_tokens"] == 0
    assert settings["max_new_tokens"] == GENERATION_MAX_NEW_TOKENS == 512
    assert settings["do_sample"] is True
    assert settings["temperature"] == 0.8


def test_greedy_differs_from_the_samples_only_in_sampling():
    greedy = frozen_decoding_settings(temperature=0.0)
    samples = frozen_decoding_settings(temperature=0.8)

    differing = {key for key in samples if greedy[key] != samples[key]}
    assert differing == {"do_sample", "temperature"}
    assert greedy["do_sample"] is False
    # Sampling off, so the recorded temperature is a true no-op rather than a
    # value transformers silently ignores.
    assert greedy["temperature"] == 1.0
    assert greedy["repetition_penalty"] == 1.0


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_checkpoint_values_cannot_override_the_frozen_config(temperature):
    ours = build_generation_config(temperature=temperature)
    checkpoint = _checkpoint_generation_config()
    # The fixture really does carry the values §4.4 warns about.
    assert (checkpoint.top_p, checkpoint.top_k, checkpoint.repetition_penalty) == (0.8, 20, 1.05)

    merged = _merge_like_transformers(ours, checkpoint)

    assert merged.top_p == 1.0
    assert merged.top_k == 0
    assert merged.repetition_penalty == 1.0
    assert merged.max_new_tokens == GENERATION_MAX_NEW_TOKENS
    assert merged.temperature == (1.0 if temperature == 0.0 else temperature)
    assert merged.do_sample is (temperature != 0.0)


def test_unpinnable_checkpoint_knobs_leak_unless_the_checkpoint_config_is_emptied():
    """Why `_neutralize_checkpoint_generation_config` exists: a knob whose only
    neutral value is `None` cannot be pinned, because transformers reads `None`
    as "unset" and refills it from the checkpoint."""
    merged = _merge_like_transformers(
        build_generation_config(temperature=0.8), _checkpoint_generation_config()
    )
    assert merged.penalty_alpha == 0.6
    assert merged.bad_words_ids == [[13]]


def test_adapter_empties_the_checkpoint_config_but_keeps_its_stop_tokens():
    model = _RecordingModel([1], generation_config=_checkpoint_generation_config())
    adapter = _RealModelAdapter(model, _StubTokenizer())

    neutralized = model.generation_config
    assert neutralized.top_k is None
    assert neutralized.top_p is None
    assert neutralized.repetition_penalty is None
    assert neutralized.penalty_alpha is None
    assert neutralized.bad_words_ids is None
    # Stop/pad tokens are model identity, not decoding policy — dropping them
    # would stop generation from ever terminating before the 512-token cap.
    assert neutralized.eos_token_id == [151645, 151643]
    assert neutralized.pad_token_id == 151643
    assert adapter.eos_token_ids == frozenset({151645, 151643, 7})
    # The original is kept for the record, not for generation.
    assert adapter.checkpoint_generation_config.top_k == 20


def test_merge_against_the_emptied_config_leaves_nothing_of_the_checkpoint():
    model = _RecordingModel([1], generation_config=_checkpoint_generation_config())
    _RealModelAdapter(model, _StubTokenizer())

    merged = _merge_like_transformers(
        build_generation_config(temperature=0.8), model.generation_config
    )

    assert merged.top_p == 1.0
    assert merged.top_k == 0
    assert merged.repetition_penalty == 1.0
    assert merged.penalty_alpha is None
    assert merged.bad_words_ids is None
    assert merged.eos_token_id == [151645, 151643]


def test_generate_passes_the_frozen_config_and_no_loose_decoding_kwargs(monkeypatch):
    _install_stub_torch(monkeypatch)
    model = _RecordingModel([65, 66])
    adapter = _RealModelAdapter(model, _StubTokenizer(), max_new_tokens=8)

    adapter.generate("item-1", "hi", temperature=0.8, sample_id=3)

    assert model.received_config.top_p == 1.0
    assert model.received_config.top_k == 0
    assert model.received_config.repetition_penalty == 1.0
    assert model.received_config.do_sample is True
    assert model.received_config.temperature == 0.8
    assert model.received_config.max_new_tokens == 8
    # Only the model input travels as a loose kwarg; every decoding setting
    # rides in the config object, which is what makes it win the merge.
    assert set(model.received_kwargs) == {"attention_mask"}


# --- (b) stored log-probabilities are raw ------------------------------------


def test_generate_stores_raw_logprobs_from_outputs_logits(monkeypatch):
    _install_stub_torch(monkeypatch)
    model = _RecordingModel([3, 5])
    adapter = _RealModelAdapter(model, _StubTokenizer(), max_new_tokens=8)

    sample = adapter.generate("item-1", "hi", temperature=0.0, sample_id=0)

    # Raw logits give the chosen token 0.0 against fifteen -10.0 entries.
    expected = -math.log(1.0 + 15.0 * math.exp(-10.0))
    assert sample.token_logprobs == pytest.approx([expected, expected])
    # The uniform "processed" scores would have given log(1/16) instead —
    # confirming the assertion above can actually tell the two fields apart.
    assert expected != pytest.approx(-math.log(16.0))
    assert sample.token_ids == [3, 5]


def test_frozen_settings_request_logits_and_not_scores():
    settings = frozen_decoding_settings(temperature=0.8)
    assert settings["output_logits"] is True
    assert settings["output_scores"] is False
    assert settings["return_dict_in_generate"] is True


# --- (c) truncation record ---------------------------------------------------


def test_hit_length_cap_only_when_the_cap_is_reached_without_a_stop_token():
    eos = frozenset({7})
    assert hit_length_cap([1, 2, 3], 8, eos) is False          # stopped early
    assert hit_length_cap([1] * 8, 8, eos) is True             # ran into the cap
    assert hit_length_cap([1] * 7 + [7], 8, eos) is False      # EOS exactly at the cap
    assert hit_length_cap([1] * 9, 8, eos) is True             # defensive: over the cap


def test_generate_flags_a_generation_that_stopped_at_the_cap(monkeypatch):
    _install_stub_torch(monkeypatch)
    adapter = _RealModelAdapter(_RecordingModel([1, 2, 3]), _StubTokenizer(), max_new_tokens=3)
    assert adapter.generate("item", "hi", temperature=0.0, sample_id=0).truncated_at_cap is True


def test_generate_does_not_flag_a_generation_that_stopped_on_its_own(monkeypatch):
    _install_stub_torch(monkeypatch)
    adapter = _RealModelAdapter(_RecordingModel([1, 2]), _StubTokenizer(), max_new_tokens=8)
    assert adapter.generate("item", "hi", temperature=0.0, sample_id=0).truncated_at_cap is False


def test_generate_does_not_flag_eos_arriving_exactly_at_the_cap(monkeypatch):
    _install_stub_torch(monkeypatch)
    adapter = _RealModelAdapter(_RecordingModel([1, 2, 7]), _StubTokenizer(), max_new_tokens=3)
    assert adapter.generate("item", "hi", temperature=0.0, sample_id=0).truncated_at_cap is False


# --- (e) the manifest record -------------------------------------------------


def test_resolved_record_carries_our_values_and_the_library_defaults():
    record = resolved_decoding_settings(temperature=0.8)

    assert record["follows_checkpoint_generation_config"] is False
    assert record["requested_temperature"] == 0.8
    resolved = record["resolved"]
    # Ours.
    assert resolved["top_p"] == 1.0
    assert resolved["top_k"] == 0
    assert resolved["repetition_penalty"] == 1.0
    assert resolved["max_new_tokens"] == GENERATION_MAX_NEW_TOKENS
    assert resolved["output_logits"] is True
    # Left to the library, and recorded at its resolved value rather than as a
    # blank — this is the half §4.4's "record ... decoding settings" would
    # otherwise miss.
    assert "num_beams" not in record["explicitly_set"] or resolved["num_beams"] == 1
    assert resolved["min_p"] is None
    assert resolved["suppress_tokens"] is None
    assert resolved["bad_words_ids"] is None
    assert resolved["no_repeat_ngram_size"] == 0


def test_decoding_settings_id_separates_greedy_from_samples_and_is_stable():
    greedy = resolved_decoding_settings(temperature=0.0)
    samples = resolved_decoding_settings(temperature=0.8)

    assert decoding_settings_id(greedy) == decoding_settings_id(
        resolved_decoding_settings(temperature=0.0)
    )
    assert decoding_settings_id(greedy) != decoding_settings_id(samples)
