"""Shared orchestration core for the real (non-mock) H100 drivers —
`scripts/run_pilot.py` (PILOT_MODELS, BNB-nf4 arm first per §4.7) and
`scripts/run_main.py` (MAIN_ANALYSIS_MODELS, full quantization ladder).
Both are thin CLI wrappers over `run()` here, selecting which models/quant
levels/item scope to use.

Structurally mirrors `qcd.dry_run`'s generate -> score -> detect -> write
loop, but against `load_model(mock=False)` and the real dataset loaders.
Every step here already works end-to-end **except the one GPU-dependent
link in the chain**: `models/loader.py`'s real `generate()`/
`score_logprobs()` bodies still raise `NotImplementedError` (by design —
deferred to a future session on a CUDA-capable machine, see
`pipeline_build_plan.md`). Running this against a real model surfaces that
`NotImplementedError` at exactly the point generation is first attempted,
after items/manifest are already written to disk — this module doesn't
paper over the gap.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections import defaultdict
from pathlib import Path

from qcd.config import ModelSpec, Quant
from qcd.constants import CDD_N_SAMPLES, CDD_SAMPLE_TEMPERATURE
from qcd.data.humaneval import load_humaneval
from qcd.data.livecodebench import load_livecodebench_split
from qcd.data.mbppplus import load_mbppplus
from qcd.data.schema import Dataset, Item
from qcd.detectors.cdd import peakedness
from qcd.detectors.mink_prob import mink_prob
from qcd.detectors.perplexity import negative_log_perplexity_score
from qcd.generation.cache import GenerationCache
from qcd.generation.sampler import sample_item
from qcd.io.manifest import build_manifest, write_manifest
from qcd.io.raw_writer import RawDataWriter
from qcd.models.loader import load_model
from qcd.scoring.pass_rate import partial_pass_rate

_EVALPLUS_DATASETS = (Dataset.HUMANEVAL, Dataset.MBPPPLUS)


@dataclasses.dataclass
class RealRunConfig:
    models: tuple[ModelSpec, ...]
    quant_levels: tuple[Quant, ...]
    output_dir: Path
    lcb_cutoff_boundary: dt.datetime
    lcb_release_version: str = "release_v6"
    n_cdd_samples: int = CDD_N_SAMPLES
    include_humaneval: bool = True
    include_mbppplus: bool = True
    item_limit_per_condition: int | None = None  # pilot-scale cap; None = all items


def load_all_items(config: RealRunConfig) -> list[Item]:
    lcb_pre, lcb_post = load_livecodebench_split(config.lcb_cutoff_boundary, release_version=config.lcb_release_version)
    items: list[Item] = list(lcb_pre) + list(lcb_post)
    if config.include_humaneval:
        items += load_humaneval()
    if config.include_mbppplus:
        items += load_mbppplus()

    if config.item_limit_per_condition is not None:
        counts: dict[Dataset, int] = defaultdict(int)
        capped = []
        for item in items:
            if counts[item.dataset] < config.item_limit_per_condition:
                capped.append(item)
                counts[item.dataset] += 1
        items = capped
    return items


def _assemble_candidate_code(item: Item, completion_text: str) -> str:
    """HumanEval+/MBPP+ prompts are Python code prefixes the completion
    continues (evalplus's own `prompt + canonical_solution` convention,
    scoring/sandbox.py's docstring). LiveCodeBench prompts are natural-
    language problem statements — the completion is assumed to already be
    complete, runnable source with no markdown fencing. Extracting a code
    block out of a raw chat-style model response (```python ... ```) is a
    known gap, not handled here — flagged rather than silently assumed."""
    if item.dataset in _EVALPLUS_DATASETS:
        return item.prompt + completion_text
    return completion_text


def run(config: RealRunConfig) -> None:
    items = load_all_items(config)

    writer = RawDataWriter(config.output_dir / "raw")
    writer.write_items(items)

    manifest = build_manifest(
        {
            "models": [m.name for m in config.models],
            "quant_levels": [q.value for q in config.quant_levels],
            "n_items": len(items),
            "lcb_cutoff_boundary": config.lcb_cutoff_boundary.isoformat(),
            "lcb_release_version": config.lcb_release_version,
        }
    )
    write_manifest(manifest, config.output_dir / "manifest.json")

    cache = GenerationCache(config.output_dir / "cache")

    for model_spec in config.models:
        for quant in config.quant_levels:
            model = load_model(model_spec, quant, mock=False)  # NotImplementedError today, by design (see module docstring)
            for item in items:
                generations = sample_item(
                    model, cache, model_name=model_spec.name, quant=quant.value,
                    item_id=item.item_id, prompt=item.prompt, n_samples=config.n_cdd_samples,
                )
                candidate_code = _assemble_candidate_code(item, generations.greedy.text)
                pass_rate = partial_pass_rate(item, candidate_code)

                writer.add_generation(
                    model=model_spec.name, quant=quant.value, item_id=item.item_id, sample_id=0, is_greedy=True,
                    text=generations.greedy.text, token_ids=generations.greedy.token_ids,
                    token_logprobs=generations.greedy.token_logprobs, partial_pass_rate=pass_rate,
                    decoding_temperature=0.0,
                )
                for sample_id, sample in enumerate(generations.samples, start=1):
                    writer.add_generation(
                        model=model_spec.name, quant=quant.value, item_id=item.item_id, sample_id=sample_id,
                        is_greedy=False, text=sample.text, token_ids=sample.token_ids,
                        token_logprobs=sample.token_logprobs, decoding_temperature=CDD_SAMPLE_TEMPERATURE,
                    )

                cdd_score = peakedness(generations.greedy.token_ids, [s.token_ids for s in generations.samples])
                ppl_score = negative_log_perplexity_score(generations.greedy.token_logprobs)
                mink_score = mink_prob(generations.greedy.token_logprobs)
                for detector, score in (("cdd", cdd_score), ("perplexity", ppl_score), ("mink_prob", mink_score)):
                    writer.add_detector_score(model=model_spec.name, quant=quant.value, item_id=item.item_id, detector=detector, score=score)

    writer.flush()
