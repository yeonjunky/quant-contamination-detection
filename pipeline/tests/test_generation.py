import tempfile
from pathlib import Path

from qcd.generation.cache import CacheKey, GenerationCache
from qcd.generation.sampler import sample_item
from qcd.models.mock import MockModel


def _cache(tmp_path: Path) -> GenerationCache:
    return GenerationCache(tmp_path / "cache")


def test_cache_roundtrip(tmp_path):
    cache = _cache(tmp_path)
    key = CacheKey(model_name="m", quant="bf16", item_id="x", is_greedy=True, sample_ids=(0,), prompt="p")
    assert cache.get(key) is None
    assert key not in cache

    cache.put(key, {"text": "hello"})
    assert key in cache
    assert cache.get(key) == {"text": "hello"}


def test_cache_key_digest_changes_with_prompt():
    k1 = CacheKey(model_name="m", quant="bf16", item_id="x", is_greedy=True, sample_ids=(0,), prompt="prompt A")
    k2 = CacheKey(model_name="m", quant="bf16", item_id="x", is_greedy=True, sample_ids=(0,), prompt="prompt B")
    assert k1.digest != k2.digest


def test_cache_key_digest_changes_with_revision_temperature_and_generation_config():
    base = dict(
        model_name="m", quant="bf16", item_id="x", is_greedy=False,
        sample_ids=(0,), prompt="p",
    )
    original = CacheKey(
        **base, temperature=0.8, model_revision="rev-a", generation_config="max=512",
    )
    assert original.digest != CacheKey(
        **base, temperature=0.7, model_revision="rev-a", generation_config="max=512",
    ).digest
    assert original.digest != CacheKey(
        **base, temperature=0.8, model_revision="rev-b", generation_config="max=512",
    ).digest
    assert original.digest != CacheKey(
        **base, temperature=0.8, model_revision="rev-a", generation_config="max=1024",
    ).digest


def test_cache_key_digest_names_every_sample_id_in_the_entry():
    base = dict(model_name="m", quant="bf16", item_id="x", is_greedy=False, prompt="p", temperature=0.8)
    assert CacheKey(**base, sample_ids=(0, 1)).digest != CacheKey(**base, sample_ids=(0, 1, 2)).digest
    assert CacheKey(**base, sample_ids=(0, 1)).digest != CacheKey(**base, sample_ids=(1, 0)).digest


def test_sample_item_shape(tmp_path):
    model = MockModel()
    model.register_item("x", contaminated=False, quality=0.5)
    cache = _cache(tmp_path)

    result = sample_item(model, cache, model_name="mock", quant="bf16", item_id="x", prompt="p", n_samples=5)

    assert result.greedy.is_greedy is True
    assert len(result.samples) == 5
    assert all(not s.is_greedy for s in result.samples)


def test_sample_item_uses_cache_on_second_call(tmp_path):
    model = MockModel()
    model.register_item("x", contaminated=False, quality=0.5)
    cache = _cache(tmp_path)

    first = sample_item(model, cache, model_name="mock", quant="bf16", item_id="x", prompt="p", n_samples=3)
    # Second call must be servable purely from cache — deregister the item so
    # a cache-miss fallback to model.generate() would raise KeyError instead
    # of silently regenerating and masking a caching bug.
    model_after = MockModel()
    second = sample_item(model_after, cache, model_name="mock", quant="bf16", item_id="x", prompt="p", n_samples=3)

    assert first.greedy.token_ids == second.greedy.token_ids
    assert [s.token_ids for s in first.samples] == [s.token_ids for s in second.samples]


def test_greedy_generation_is_deterministic_across_runs(tmp_path):
    model = MockModel()
    model.register_item("x", contaminated=True, quality=0.9)
    cache1 = _cache(Path(tempfile.mkdtemp()))
    cache2 = _cache(Path(tempfile.mkdtemp()))

    r1 = sample_item(model, cache1, model_name="mock", quant="bf16", item_id="x", prompt="p", n_samples=2)
    r2 = sample_item(model, cache2, model_name="mock", quant="bf16", item_id="x", prompt="p", n_samples=2)

    assert r1.greedy.token_ids == r2.greedy.token_ids


class _RecordingMock(MockModel):
    def __init__(self):
        super().__init__()
        self.calls = []

    def generate(self, item_id, prompt, *, temperature, sample_id):
        self.calls.append(("generate", temperature, sample_id))
        return super().generate(item_id, prompt, temperature=temperature, sample_id=sample_id)

    def generate_samples(self, item_id, prompt, *, temperature, sample_ids):
        self.calls.append(("generate_samples", temperature, list(sample_ids)))
        return [
            MockModel.generate(self, item_id, prompt, temperature=temperature, sample_id=s)
            for s in sample_ids
        ]


def test_a_missing_sample_batch_is_generated_whole_in_one_call(tmp_path):
    model = _RecordingMock()
    model.register_item("x", contaminated=False, quality=0.5)

    result = sample_item(model, _cache(tmp_path), model_name="mock", quant="bf16", item_id="x", prompt="p", n_samples=4)

    assert model.calls == [("generate", 0.0, 0), ("generate_samples", 0.8, [0, 1, 2, 3])]
    assert len(result.samples) == 4


def test_a_cached_sample_batch_is_served_without_calling_the_model(tmp_path):
    cache = _cache(tmp_path)
    first_model = _RecordingMock()
    first_model.register_item("x", contaminated=False, quality=0.5)
    first = sample_item(first_model, cache, model_name="mock", quant="bf16", item_id="x", prompt="p", n_samples=4)

    second_model = _RecordingMock()
    second_model.register_item("x", contaminated=False, quality=0.5)
    second = sample_item(second_model, cache, model_name="mock", quant="bf16", item_id="x", prompt="p", n_samples=4)

    assert second_model.calls == []
    assert [s.token_ids for s in second.samples] == [s.token_ids for s in first.samples]

