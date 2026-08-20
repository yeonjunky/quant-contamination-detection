"""Shared orchestration core for the real (non-mock) H100 drivers —
`scripts/run_pilot.py` (PILOT_MODELS, BNB-nf4 arm first per §4.7) and
`scripts/run_main.py` (MAIN_ANALYSIS_MODELS, full quantization ladder).
Both are thin CLI wrappers over `run()` here, selecting which models/quant
levels/item scope to use.

Structurally mirrors `qcd.dry_run`'s generate -> score -> detect -> write
loop, but against `load_model(mock=False)` and the real dataset loaders.
The real bf16/bnb/AWQ model paths and fixed-prompt probability scoring are
implemented and H100-smoke-tested. Generated-completion log-probabilities
remain stored as exploratory confidence data; Q1's perplexity/Min-k scores
come from identical fixed benchmark prompts at every precision.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import re
import time
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
from qcd.scoring.logprob import score_prompt_logprobs
from qcd.scoring.pass_rate import partial_pass_rate

_EVALPLUS_DATASETS = (Dataset.HUMANEVAL, Dataset.MBPPPLUS)

# Matches a ```-fenced block, optional language tag (```python, ```py, bare
# ```, ...). DOTALL so the fence content can span multiple lines.
_CODE_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n?(.*?)```", re.DOTALL)


def _strip_markdown_fence(text: str) -> str:
    """If `text` contains one or more ```-fenced blocks, return the content
    of the LAST one (a chat model sometimes shows an earlier/wrong attempt
    before settling on a final answer); otherwise return `text` unchanged
    (a model can answer with a bare, un-fenced continuation).

    Run before evalplus's sanitize()/code_extract() rather than relying on
    those alone: evalplus's `code_extract` searches for the longest
    syntactically-valid *contiguous* line range, but the fence delimiter
    lines themselves (` ```python `, ` ``` `) are never valid Python, so a
    short, single-line fenced answer surrounded by prose can end up with no
    valid ≥2-line window at all — confirmed empirically (a
    `print('hi')`-only LCB-style completion silently fell back to returning
    the completion's first line, unrelated prose, with no signal that
    extraction had failed). Stripping the fence first removes the
    delimiters that caused that, and evalplus's extractors still run
    afterward as a backstop for any residual prose inside or around the
    fence."""
    matches = _CODE_FENCE_RE.findall(text)
    return matches[-1] if matches else text


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
    """The roster is entirely -Instruct models queried through a chat
    template (models/loader.py's `_RealModelAdapter._build_input_ids`), so
    `completion_text` is chat-style output — prose plus a markdown code
    fence, not a raw code continuation — for every dataset, not just
    LiveCodeBench. Confirmed on real HumanEval output during the 2026-08-15
    real-GPU smoke test (pipeline_implementation_log.md): concatenating
    `item.prompt + completion_text` verbatim produced unrunnable code and a
    0.0 partial-pass-rate on all 5 real items; extracting first raised 4/5
    to 1.0 (the 5th's near-zero score was a genuine model logic error, not
    an extraction failure).

    First strips a ```-fenced block via `_strip_markdown_fence` (see its
    docstring for why that has to happen before, not instead of, the next
    step), then reuses evalplus's own post-processing (`evalplus/
    sanitize.py` — the same step evalplus's own leaderboard runs on LLM
    output before scoring) rather than reimplementing the rest:
    - HumanEval+/MBPP+: `sanitize(prompt + completion, entrypoint=...)` —
      exactly `evalplus.sanitize.script()`'s own recipe for instruct-model
      output. AST-extracts (tree-sitter) only the definitions reachable from
      the target function, discarding prose and any extraneous code.
      Empirically validated on 5 real HumanEval completions (2026-08-15 real
      GPU smoke test): 4/5 went from 0.0 (no extraction) to 1.0 partial-pass
      after this fix; the 5th's near-zero score was a genuine model logic
      error, not an extraction failure.
    - LiveCodeBench: `code_extract(completion)` alone, no entrypoint.
      `sanitize()`'s AST path only preserves import/class/function/
      assignment nodes — an LCB stdin-style candidate is often a plain
      imperative script (bare `input()`/`print()` calls, no wrapping
      function), which that path would silently drop. `code_extract`'s
      "longest syntactically valid contiguous line range" has no such
      requirement, so it's the safer of evalplus's two extractors for LCB's
      more varied shapes (stdin scripts and `class Solution` alike). Not
      empirically validated against a real LCB completion in this session
      (only HumanEval was exercised on real hardware) — worth confirming
      once a real LCB generation is available.
    """
    from evalplus.sanitize import code_extract, sanitize  # noqa: PLC0415

    fenced = _strip_markdown_fence(completion_text)
    if item.dataset in _EVALPLUS_DATASETS:
        entry_point = item.metadata["evalplus_problem"]["entry_point"]
        return sanitize(item.prompt + fenced, entrypoint=entry_point)
    return code_extract(fenced)


def run(config: RealRunConfig) -> None:
    items = load_all_items(config)

    writer = RawDataWriter(config.output_dir / "raw")
    writer.write_items(items)

    manifest = build_manifest(
        {
            "models": [m.name for m in config.models],
            "quant_levels": [q.value for q in config.quant_levels],
            "baseline_dtype": "bfloat16",
            "n_items": len(items),
            "lcb_cutoff_boundary": config.lcb_cutoff_boundary.isoformat(),
            "lcb_release_version": config.lcb_release_version,
        }
    )
    write_manifest(manifest, config.output_dir / "manifest.json")

    cache = GenerationCache(config.output_dir / "cache")

    for model_spec in config.models:
        for quant in config.quant_levels:
            model = load_model(model_spec, quant, mock=False)
            model_config = getattr(getattr(model, "model", None), "config", None)
            model_revision = getattr(model_config, "_commit_hash", None)
            tokenizer_revision = getattr(
                getattr(model, "tokenizer", None), "init_kwargs", {}
            ).get("_commit_hash")
            for item in items:
                started = time.perf_counter()
                generations = sample_item(
                    model, cache, model_name=model_spec.name, quant=quant.value,
                    item_id=item.item_id, prompt=item.prompt, n_samples=config.n_cdd_samples,
                )
                generation_seconds = time.perf_counter() - started
                candidate_code = _assemble_candidate_code(item, generations.greedy.text)
                started = time.perf_counter()
                pass_rate = partial_pass_rate(item, candidate_code)
                sandbox_scoring_seconds = time.perf_counter() - started
                started = time.perf_counter()
                prompt_logprobs = score_prompt_logprobs(
                    model, item.item_id, item.prompt
                )
                prompt_scoring_seconds = time.perf_counter() - started

                writer.add_generation(
                    model=model_spec.name, quant=quant.value, item_id=item.item_id, sample_id=0, is_greedy=True,
                    text=generations.greedy.text, token_ids=generations.greedy.token_ids,
                    token_logprobs=generations.greedy.token_logprobs, partial_pass_rate=pass_rate,
                    passed=bool(pass_rate == 1.0),
                    prompt_token_logprobs=prompt_logprobs,
                    decoding_temperature=0.0,
                    generation_seconds=generation_seconds,
                    prompt_scoring_seconds=prompt_scoring_seconds,
                    sandbox_scoring_seconds=sandbox_scoring_seconds,
                    model_revision=model_revision,
                    tokenizer_revision=tokenizer_revision,
                )
                for sample_id, sample in enumerate(generations.samples, start=1):
                    writer.add_generation(
                        model=model_spec.name, quant=quant.value, item_id=item.item_id, sample_id=sample_id,
                        is_greedy=False, text=sample.text, token_ids=sample.token_ids,
                        token_logprobs=sample.token_logprobs, decoding_temperature=CDD_SAMPLE_TEMPERATURE,
                        model_revision=model_revision,
                        tokenizer_revision=tokenizer_revision,
                    )

                cdd_score = peakedness(generations.greedy.token_ids, [s.token_ids for s in generations.samples])
                ppl_score = negative_log_perplexity_score(prompt_logprobs)
                mink_score = mink_prob(prompt_logprobs)
                completion_ppl_score = negative_log_perplexity_score(
                    generations.greedy.token_logprobs
                )
                completion_mink_score = mink_prob(generations.greedy.token_logprobs)
                for detector, score in (
                    ("cdd", cdd_score),
                    ("perplexity", ppl_score),
                    ("mink_prob", mink_score),
                    ("completion_perplexity", completion_ppl_score),
                    ("completion_mink_prob", completion_mink_score),
                ):
                    writer.add_detector_score(model=model_spec.name, quant=quant.value, item_id=item.item_id, detector=detector, score=score)

    writer.flush()
