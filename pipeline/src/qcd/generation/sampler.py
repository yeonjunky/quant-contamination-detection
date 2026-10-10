"""1 greedy + n temperature-sampled generations per item, per precision
(paper §4.4's "Sampling protocol", exact constants pinned in constants.py —
`CDD_N_SAMPLES=50`, `CDD_SAMPLE_TEMPERATURE=0.8`, `CDD_GREEDY_TEMPERATURE=0.0`,
"We use n=50, matching the original paper.").

Deliberately backend-agnostic: takes any object satisfying
`models.loader.LoadedModel`'s `generate()`/`generate_samples()` call surface,
so this module runs unchanged against `MockModel` and the real backend.
Every generation is routed through `generation/cache.py` first, so the continuous-scoring pipeline (step 1) and
the detector-scoring pipeline (step 2) reuse the same underlying samples
instead of regenerating them (§4.4's cost-sharing directive).
"""

from __future__ import annotations

import dataclasses

from qcd.constants import CDD_GREEDY_TEMPERATURE, CDD_N_SAMPLES, CDD_SAMPLE_TEMPERATURE
from qcd.generation.cache import CacheKey, GenerationCache


@dataclasses.dataclass
class ItemGenerations:
    item_id: str
    greedy: object  # a GenerationSample (or backend-equivalent); untyped to avoid coupling to models.mock
    samples: list  # list of GenerationSample, length n_samples, all at sample_temperature


GREEDY_SAMPLE_ID = 0


def sample_ids(n_samples: int) -> tuple[int, ...]:
    """The ids of an item's `n_samples` temperature samples, in the order of
    `ItemGenerations.samples`. A sample's id is both the `sample_id` its raw
    row carries and the id its generator seed is derived from; the greedy
    output is id 0."""
    return tuple(range(GREEDY_SAMPLE_ID + 1, GREEDY_SAMPLE_ID + 1 + n_samples))


def sample_item(
    model,
    cache: GenerationCache,
    *,
    model_name: str,
    quant: str,
    item_id: str,
    prompt: str,
    batch_size: int,
    n_samples: int = CDD_N_SAMPLES,
    sample_temperature: float = CDD_SAMPLE_TEMPERATURE,
    greedy_temperature: float = CDD_GREEDY_TEMPERATURE,
    model_revision: str = "",
    generation_config: str = "",
) -> ItemGenerations:
    """`batch_size` is the model's fixed row count per `generate_samples`
    call. It does not enter the cache key here; the caller must put it in
    `generation_config` so samples drawn with different batch sizes never
    share an entry."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    def key(*, is_greedy: bool, sample_ids: tuple[int, ...], temperature: float) -> CacheKey:
        return CacheKey(
            model_name=model_name, quant=quant, item_id=item_id,
            is_greedy=is_greedy, sample_ids=sample_ids, prompt=prompt,
            temperature=temperature, model_revision=model_revision,
            generation_config=generation_config,
        )

    greedy = _get_or_generate(
        cache, key(is_greedy=True, sample_ids=(GREEDY_SAMPLE_ID,), temperature=greedy_temperature),
        lambda: model.generate(
            item_id, prompt, temperature=greedy_temperature, sample_id=GREEDY_SAMPLE_ID,
        ),
    )
    # A real backend's logits depend slightly on which rows share a batch, so
    # the chunks are fixed by (n_samples, batch_size) alone, and the item's
    # whole sample set is one cache entry that is always regenerated whole.
    ids = sample_ids(n_samples)
    samples = _get_or_generate(
        cache, key(is_greedy=False, sample_ids=ids, temperature=sample_temperature),
        lambda: [
            sample
            for start in range(0, n_samples, batch_size)
            for sample in model.generate_samples(
                item_id, prompt, temperature=sample_temperature,
                sample_ids=list(ids[start:start + batch_size]),
            )
        ],
    )
    return ItemGenerations(item_id=item_id, greedy=greedy, samples=samples)


def _get_or_generate(cache: GenerationCache, key: CacheKey, generate):
    cached = cache.get(key)
    if cached is not None:
        return cached
    generated = generate()
    cache.put(key, generated)
    return generated
