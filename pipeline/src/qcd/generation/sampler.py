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


def sample_item(
    model,
    cache: GenerationCache,
    *,
    model_name: str,
    quant: str,
    item_id: str,
    prompt: str,
    n_samples: int = CDD_N_SAMPLES,
    sample_temperature: float = CDD_SAMPLE_TEMPERATURE,
    greedy_temperature: float = CDD_GREEDY_TEMPERATURE,
    model_revision: str = "",
    generation_config: str = "",
) -> ItemGenerations:
    def key(*, is_greedy: bool, sample_ids: tuple[int, ...], temperature: float) -> CacheKey:
        return CacheKey(
            model_name=model_name, quant=quant, item_id=item_id,
            is_greedy=is_greedy, sample_ids=sample_ids, prompt=prompt,
            temperature=temperature, model_revision=model_revision,
            generation_config=generation_config,
        )

    greedy = _get_or_generate(
        cache, key(is_greedy=True, sample_ids=(0,), temperature=greedy_temperature),
        lambda: model.generate(item_id, prompt, temperature=greedy_temperature, sample_id=0),
    )
    # The whole sample batch is one cache entry and one generate call: a real
    # backend's logits depend slightly on batch size, so regenerating only the
    # missing samples of an item would not reproduce an uninterrupted run.
    sample_ids = tuple(range(n_samples))
    samples = _get_or_generate(
        cache, key(is_greedy=False, sample_ids=sample_ids, temperature=sample_temperature),
        lambda: model.generate_samples(
            item_id, prompt, temperature=sample_temperature, sample_ids=list(sample_ids)
        ),
    )
    return ItemGenerations(item_id=item_id, greedy=greedy, samples=samples)


def _get_or_generate(cache: GenerationCache, key: CacheKey, generate):
    cached = cache.get(key)
    if cached is not None:
        return cached
    generated = generate()
    cache.put(key, generated)
    return generated
