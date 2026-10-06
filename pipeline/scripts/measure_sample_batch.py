#!/usr/bin/env python
"""Measure which CDD sample batch size fits one model on this GPU.

`models/registry.py` gives each model a fixed `sample_batch_size`, the row
count of every batched `generate_samples` call (paper §4.4). The 32B models
have none yet, and the main run refuses to start until they do. This script
produces the measurement a person needs to set it. It never writes the
registry: a person reads the record, sets the value, and commits it.

For one `--model`/`--quant`, the model is loaded once. The N LiveCodeBench
release_v6 items with the longest generation prompts are chosen, counting
tokens the way the model will see them (`real_run._generation_prompt`
rendered through the adapter's chat template). Then, for each candidate batch
size from largest to smallest, every chosen item goes through the production
`sample_item` path with n=50 samples into a throwaway cache. Per candidate it
records whether CUDA ran out of memory, peak allocated and reserved memory,
seconds per item (greedy plus 50 samples), and the generated-length
distribution with the 512-token cap rate. A candidate that runs out of memory
is recorded and the smaller ones are still tried.

Reproducibility. At each candidate that fits, the longest item is generated a
second time into a fresh cache, and every token id and log-probability of the
greedy output and all 50 samples must match the first run exactly. The main
run relies on this: a resumed item must come out the same as if it had never
been interrupted.

Safety margin rule. A candidate is acceptable when every item and the repeat
ran without running out of memory, the repeat was identical, and peak
*reserved* memory was at most 0.90 of the card's total memory. The printed
recommendation is the largest acceptable candidate. The 10% is headroom for
what this measurement does not cover: allocator fragmentation over a run of
thousands of items, and items whose prompt plus 512 generated tokens is longer
than any measured here. `worst_case_length_reached` says whether any measured
generation actually hit the 512-token cap; when it is false, the measured peak
is below the true worst case.

**Engineering validation (paper §4.6).** Output goes to
`data/raw/validation/sample_batch_measurement/` with a manifest recording
`study_phase="engineering_validation"`. Only memory, time, generated length
and run-to-run identity are recorded. No pass rate, detector score, AUC or
effect size is computed, and no generated text is kept.

Usage:
  python scripts/measure_sample_batch.py --model Qwen2.5-32B-Instruct --quant bf16
  python scripts/measure_sample_batch.py --model Olmo3.1-32B-Instruct --quant bnb_nf4 \\
      --batch-sizes 50 25 10 5 --n-items 3
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from qcd.config import Quant
from qcd.constants import CDD_N_SAMPLES, LCB_SHARED_CONTROL_BOUNDARY
from qcd.data.livecodebench import REPO_REVISION as LCB_REPO_REVISION, load_livecodebench_split
from qcd.generation.cache import GenerationCache
from qcd.generation.sampler import sample_item
from qcd.io.manifest import StudyPhase, build_manifest, write_manifest
from qcd.models.registry import get_model
from qcd.real_run import _generation_prompt

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT = _REPO_ROOT / "data" / "raw" / "validation" / "sample_batch_measurement"
_QUANT_CHOICES = tuple(quant.value for quant in Quant)
LCB_RELEASE = "release_v6"
SAFETY_FRACTION = 0.90


@dataclasses.dataclass
class CandidateResult:
    batch_size: int
    out_of_memory: bool
    # The item whose generation ran out of memory; "repeat" when it was the
    # reproducibility repeat.
    out_of_memory_at: str | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    seconds_per_item: list[float] = dataclasses.field(default_factory=list)
    generated_tokens: dict | None = None
    reproducible: bool | None = None
    reproducibility_detail: str | None = None


class CudaDevice:
    """The torch.cuda calls this script needs, behind a small surface so the
    tests can pass a fake without torch."""

    def __init__(self) -> None:
        import torch  # noqa: PLC0415

        self._torch = torch

    def reset_peak(self) -> None:
        self._torch.cuda.reset_peak_memory_stats()

    def peaks(self) -> tuple[int | None, int | None]:
        return self._torch.cuda.max_memory_allocated(), self._torch.cuda.max_memory_reserved()

    def release(self) -> None:
        gc.collect()
        self._torch.cuda.empty_cache()

    def is_out_of_memory(self, error: BaseException) -> bool:
        return isinstance(error, self._torch.cuda.OutOfMemoryError)

    def describe(self) -> dict:
        properties = self._torch.cuda.get_device_properties(0)
        return {"name": properties.name, "total_memory_bytes": properties.total_memory}


def longest_prompts(items, count_tokens: Callable[[str], int], n: int) -> list[tuple[str, str, int]]:
    """(item_id, generation prompt, prompt tokens) for the n longest prompts."""
    counted = []
    for item in items:
        prompt = _generation_prompt(item)
        counted.append((item.item_id, prompt, count_tokens(prompt)))
    return sorted(counted, key=lambda row: (-row[2], row[0]))[:n]


def _generate(model, *, model_name: str, quant: str, item_id: str, prompt: str,
              batch_size: int, n_samples: int, cache_root: Path):
    cache = GenerationCache(tempfile.mkdtemp(dir=cache_root))
    return sample_item(
        model, cache, model_name=model_name, quant=quant, item_id=item_id, prompt=prompt,
        batch_size=batch_size, n_samples=n_samples,
        generation_config=f"batch-size-measurement;sample_batch_size={batch_size}",
    )


def _all_generations(generations) -> list:
    return [generations.greedy, *generations.samples]


def compare_runs(first, second) -> tuple[bool, str]:
    """Exact equality of token ids and log-probabilities, row by row."""
    rows = zip(_all_generations(first), _all_generations(second), strict=True)
    for index, (a, b) in enumerate(rows):
        label = "greedy" if index == 0 else f"sample {index - 1}"
        if a.token_ids != b.token_ids:
            return False, f"{label}: token ids differ"
        if a.token_logprobs != b.token_logprobs:
            largest = max(abs(x - y) for x, y in zip(a.token_logprobs, b.token_logprobs))
            return False, f"{label}: log-probabilities differ (largest difference {largest:.3g})"
    return True, f"identical over {index + 1} generations"


def summarize_lengths(generations_per_item) -> dict:
    """Generated-token counts over every greedy and sampled generation."""
    lengths, at_cap = [], 0
    for generations in generations_per_item:
        for generation in _all_generations(generations):
            lengths.append(len(generation.token_ids))
            at_cap += bool(generation.truncated_at_cap)
    ordered = sorted(lengths)
    return {
        "n_generations": len(lengths),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
        "n_at_512_cap": at_cap,
        "cap_rate": at_cap / len(lengths),
    }


def measure_candidate(model, prompts, batch_size: int, device, *, model_name: str,
                      quant: str, n_samples: int, cache_root: Path) -> CandidateResult:
    result = CandidateResult(batch_size=batch_size, out_of_memory=False)
    generations_per_item = []
    location = None
    device.release()
    device.reset_peak()
    try:
        for item_id, prompt, _ in prompts:
            location = item_id
            started = time.perf_counter()
            generations_per_item.append(_generate(
                model, model_name=model_name, quant=quant, item_id=item_id, prompt=prompt,
                batch_size=batch_size, n_samples=n_samples, cache_root=cache_root,
            ))
            result.seconds_per_item.append(time.perf_counter() - started)
        location = "repeat"
        item_id, prompt, _ = prompts[0]
        repeat = _generate(
            model, model_name=model_name, quant=quant, item_id=item_id, prompt=prompt,
            batch_size=batch_size, n_samples=n_samples, cache_root=cache_root,
        )
    except Exception as error:
        if not device.is_out_of_memory(error):
            raise
        result.out_of_memory = True
        result.out_of_memory_at = location
    if result.out_of_memory:
        # Released outside the except block, once the traceback that holds
        # the failed call's tensors is gone.
        generations_per_item.clear()
        device.release()
        return result

    result.peak_allocated_bytes, result.peak_reserved_bytes = device.peaks()
    result.generated_tokens = summarize_lengths(generations_per_item)
    result.reproducible, result.reproducibility_detail = compare_runs(generations_per_item[0], repeat)
    return result


def recommend(results: list[CandidateResult], total_memory_bytes: int | None) -> dict:
    rule = (
        f"largest batch size with no out-of-memory, an identical repeat, and peak reserved "
        f"memory <= {SAFETY_FRACTION:.2f} x total device memory"
    )
    worst_case_length_reached = any(
        r.generated_tokens["n_at_512_cap"] > 0 for r in results if r.generated_tokens
    )
    if total_memory_bytes is None:
        return {"rule": rule, "batch_size": None, "reason": "device total memory unknown",
                "worst_case_length_reached": worst_case_length_reached}
    limit = SAFETY_FRACTION * total_memory_bytes
    acceptable = [
        r.batch_size for r in results
        if not r.out_of_memory and r.reproducible and r.peak_reserved_bytes <= limit
    ]
    return {
        "rule": rule,
        "batch_size": max(acceptable) if acceptable else None,
        "reason": None if acceptable else "no candidate met the rule",
        "worst_case_length_reached": worst_case_length_reached,
    }


def _gb(value: int | None) -> str:
    return "-" if value is None else f"{value / 1e9:.1f}"


def print_table(results: list[CandidateResult]) -> None:
    print(f"{'batch':>5} {'outcome':<14} {'alloc GB':>8} {'reserved GB':>11} {'s/item':>7} "
          f"{'median tok':>10} {'cap rate':>8}  repeat")
    for r in results:
        if r.out_of_memory:
            print(f"{r.batch_size:>5} {'OOM at ' + r.out_of_memory_at:<14}")
            continue
        print(
            f"{r.batch_size:>5} {'completed':<14} {_gb(r.peak_allocated_bytes):>8} "
            f"{_gb(r.peak_reserved_bytes):>11} {statistics.fmean(r.seconds_per_item):>7.1f} "
            f"{r.generated_tokens['median']:>10} {r.generated_tokens['cap_rate']:>8.2f}  "
            f"{'identical' if r.reproducible else 'DIFFERS'} ({r.reproducibility_detail})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="ModelSpec.name from models/registry.py")
    parser.add_argument("--quant", choices=_QUANT_CHOICES, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[50, 25, 10, 5])
    parser.add_argument("--n-items", type=int, default=3,
                        help="How many of the longest LiveCodeBench prompts to measure (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: data/raw/validation/sample_batch_measurement/<model>-<quant>")
    return parser


def _load_lcb_items():
    pre, post = load_livecodebench_split(
        dt.datetime.fromisoformat(LCB_SHARED_CONTROL_BOUNDARY), release_version=LCB_RELEASE,
    )
    return [*pre, *post]


def _load_real_model(spec, quant):
    from qcd.models.loader import load_model  # noqa: PLC0415

    return load_model(spec, quant, mock=False)


def _adapter_token_count(model) -> Callable[[str], int]:
    return lambda prompt: int(model._build_input_ids(prompt)[0].shape[-1])


def _save_pip_freeze(output_dir: Path) -> Path:
    path = output_dir / "pip-freeze.txt"
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    path.write_text(freeze.stdout)
    return path


def main(argv=None, *, load_model_fn=_load_real_model, load_items=_load_lcb_items,
         device=None, count_tokens_for=_adapter_token_count,
         save_pip_freeze=_save_pip_freeze) -> Path:
    """The keyword arguments exist so tests and a CPU wiring run can inject
    a model, items and device; the command line never sets them."""
    args = build_parser().parse_args(argv)
    if any(size < 1 for size in args.batch_sizes):
        raise SystemExit(f"--batch-sizes must all be >= 1, got {args.batch_sizes}")
    if args.n_items < 1:
        raise SystemExit(f"--n-items must be >= 1, got {args.n_items}")
    spec = get_model(args.model)
    quant = Quant(args.quant)
    batch_sizes = sorted(set(args.batch_sizes), reverse=True)
    output_dir = args.output_dir or _DEFAULT_OUTPUT / f"{spec.name}-{quant.value}"
    device = device or CudaDevice()

    items = load_items()
    model = load_model_fn(spec, quant)
    prompts = longest_prompts(items, count_tokens_for(model), args.n_items)
    print(f"Batch-size measurement: model={spec.name}, quant={quant.value}, n_samples={CDD_N_SAMPLES}")
    print("  engineering validation (paper §4.6); records memory, time and generated length only")
    for item_id, _, n_tokens in prompts:
        print(f"  item {item_id}: {n_tokens} prompt tokens")

    results = []
    with tempfile.TemporaryDirectory(prefix="qcd_batch_measure_") as cache_root:
        for batch_size in batch_sizes:
            result = measure_candidate(
                model, prompts, batch_size, device, model_name=spec.name, quant=quant.value,
                n_samples=CDD_N_SAMPLES, cache_root=Path(cache_root),
            )
            results.append(result)
            print(f"  batch {batch_size}: {'out of memory' if result.out_of_memory else 'completed'}")

    gpu = device.describe()
    recommendation = recommend(results, gpu["total_memory_bytes"])
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "study_phase": StudyPhase.ENGINEERING_VALIDATION.value,
        "stores_outcome_values": False,
        "model": spec.name,
        "quant": quant.value,
        "n_samples": CDD_N_SAMPLES,
        "gpu": gpu,
        "items": [
            {"item_id": item_id, "prompt_tokens": n_tokens} for item_id, _, n_tokens in prompts
        ],
        "candidates": [dataclasses.asdict(result) for result in results],
        "recommendation": recommendation,
    }
    record_path = output_dir / "batch_measurement.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    freeze_path = save_pip_freeze(output_dir)
    manifest = build_manifest(
        {
            "driver": "scripts/measure_sample_batch.py",
            "model": spec.name,
            "model_revision": getattr(model, "revision", None) or spec.revision,
            "quant": quant.value,
            "lcb_release": LCB_RELEASE,
            "lcb_repo_revision": LCB_REPO_REVISION,
            "item_ids": [item_id for item_id, _, _ in prompts],
            "n_samples": CDD_N_SAMPLES,
            "batch_sizes": batch_sizes,
            "safety_fraction": SAFETY_FRACTION,
            "stores_outcome_values": False,
        },
        study_phase=StudyPhase.ENGINEERING_VALIDATION,
        repo_dir=_REPO_ROOT,
        extra={"gpu": gpu, "pip_freeze_path": str(freeze_path)},
    )
    write_manifest(manifest, output_dir / "manifest.json")

    print()
    print_table(results)
    print(f"\nRule: {recommendation['rule']}")
    if recommendation["batch_size"] is None:
        print(f"No recommendation: {recommendation['reason']}")
    else:
        print(f"Largest size meeting the rule: {recommendation['batch_size']} "
              f"(set it by hand in models/registry.py and commit)")
    if not recommendation["worst_case_length_reached"]:
        print("No measured generation reached the 512-token cap, so the measured peak is below "
              "the worst case.")
    print(f"Record: {record_path}")
    return record_path


if __name__ == "__main__":
    record_path = main()
    if json.loads(record_path.read_text())["recommendation"]["batch_size"] is None:
        raise SystemExit(1)
