import tempfile
from pathlib import Path

from qcd.generation.cache import CacheKey, GenerationCache
from qcd.generation.sampler import sample_item
from qcd.models.mock import MockModel


def _cache(tmp_path: Path) -> GenerationCache:
    return GenerationCache(tmp_path / "cache")


def test_cache_roundtrip(tmp_path):
    cache = _cache(tmp_path)
    key = CacheKey(model_name="m", quant="bf16", item_id="x", is_greedy=True, sample_id=0, prompt="p")
    assert cache.get(key) is None
    assert key not in cache

    cache.put(key, {"text": "hello"})
    assert key in cache
    assert cache.get(key) == {"text": "hello"}


def test_cache_key_digest_changes_with_prompt():
    k1 = CacheKey(model_name="m", quant="bf16", item_id="x", is_greedy=True, sample_id=0, prompt="prompt A")
    k2 = CacheKey(model_name="m", quant="bf16", item_id="x", is_greedy=True, sample_id=0, prompt="prompt B")
    assert k1.digest != k2.digest


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
