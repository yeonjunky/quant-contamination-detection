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
size from largest to smallest, every chosen item goes through what the main
run does per item: the production `sample_item` path with n=50 samples into a
throwaway cache, then the prompt-scoring forward pass (`score_prompt_detail`,
which holds the prompt's full float32 logits and their log-softmax). The
scores are discarded. A candidate that runs out of memory is recorded and the
smaller ones are still tried.

Each candidate runs two passes, and every number in the record says which
pass it came from:

- `normal_decoding`: the main run's decoding. Its generations give the
  reproducibility check and the digest below.
- `forced_length`: the same items with every stop token suppressed, so every
  greedy and sampled row runs to the full 512 new tokens. Its peak memory is
  the worst case a real item of this prompt length can reach. The suppression
  is a measurement-only logits processor set through the adapter's
  `extra_logits_processors` hook; the main run never sets it.

Reproducibility. Within the process, the longest item is generated a second
time into a fresh cache, and every token id and log-probability of the greedy
output and all 50 samples must match the first run exactly. The main run
relies on this: a resumed item must come out the same as if it had never been
interrupted. A resume happens in a new process, so the record also carries
`generations_sha256` per batch size: a SHA-256 over the item ids, token ids and
log-probabilities of every normal-decoding generation of the measured items,
in order. It is a fingerprint, not a score. `--compare-to <earlier record>`
compares the digests of the same cell measured in an earlier process and
exits 1 if any batch size differs.

Safety margin rule. A candidate is acceptable when both passes and the repeat
ran without running out of memory, the repeat was identical, every
forced-length row reached 512 tokens, and the forced-length peak *reserved*
memory was at most 0.90 of the card's total memory. The printed
recommendation is the largest acceptable candidate. The 10% is headroom for
what this measurement does not cover: allocator fragmentation over a run of
thousands of items, and items whose prompt is longer than any measured here.

**Engineering validation (paper §4.6).** Output goes to
`data/raw/validation/sample_batch_measurement/` with a manifest recording
`study_phase="engineering_validation"`. Only memory, time, generated length
and run-to-run identity are recorded. No pass rate, detector score, AUC or
effect size is computed. The prompt-scoring pass's log-probabilities and the
generated text are discarded unread.

Usage:
  python scripts/measure_sample_batch.py --model Qwen2.5-32B-Instruct --quant bf16
  python scripts/measure_sample_batch.py --model Olmo3.1-32B-Instruct --quant bnb_nf4 \\
      --batch-sizes 50 25 10 5 --n-items 3
  python scripts/measure_sample_batch.py --model Qwen2.5-32B-Instruct --quant bf16 \\
      --output-dir /tmp/second --compare-to \\
      ../data/raw/validation/sample_batch_measurement/Qwen2.5-32B-Instruct-bf16/batch_measurement.json
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from qcd.config import Quant
from qcd.constants import CDD_N_SAMPLES, GENERATION_MAX_NEW_TOKENS, LCB_SHARED_CONTROL_BOUNDARY
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
NORMAL = "normal_decoding"
FORCED = "forced_length"
SMALLER_SIZES_HINT = "4 3 2 1"


@dataclasses.dataclass(frozen=True)
class MeasuredItem:
    item_id: str
    # What sample_item generates from (real_run._generation_prompt).
    generation_prompt: str
    # What the main run's prompt-scoring pass scores: the item's own prompt.
    scoring_prompt: str
    prompt_tokens: int


@dataclasses.dataclass
class PassMeasurement:
    """Peak memory, time and generated length of one decoding mode."""

    decoding: str
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    seconds_per_item: list[float]
    generated_tokens: dict


@dataclasses.dataclass
class CandidateResult:
    batch_size: int
    out_of_memory: bool
    # "<decoding>:<item_id>", or "normal_decoding:repeat" for the
    # reproducibility repeat.
    out_of_memory_at: str | None = None
    normal_decoding: PassMeasurement | None = None
    forced_length: PassMeasurement | None = None
    # Every forced-length row reached the 512-token cap.
    forced_length_reached: bool | None = None
    reproducible: bool | None = None
    reproducibility_detail: str | None = None
    generations_sha256: str | None = None


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


class StopTokenSuppressor:
    """Logits processor that makes every stop token impossible, so each row
    runs to the token cap. Measurement only: it changes what is generated.

    It works on a copy, because transformers keeps the tensor it passes in as
    the step's raw logits. Duck-typed like loader._PerRowSampler."""

    def __init__(self, token_ids) -> None:
        self.token_ids = sorted(token_ids)

    def __call__(self, input_ids, scores):
        suppressed = scores.clone()
        suppressed[:, self.token_ids] = float("-inf")
        return suppressed


@contextlib.contextmanager
def forced_full_length(model):
    model.extra_logits_processors = (StopTokenSuppressor(model.eos_token_ids),)
    try:
        yield
    finally:
        model.extra_logits_processors = ()


def longest_prompts(items, count_tokens: Callable[[str], int], n: int) -> list[MeasuredItem]:
    """The n items with the longest generation prompts."""
    counted = []
    for item in items:
        prompt = _generation_prompt(item)
        counted.append(MeasuredItem(item.item_id, prompt, item.prompt, count_tokens(prompt)))
    return sorted(counted, key=lambda row: (-row.prompt_tokens, row.item_id))[:n]


def _generate(model, item: MeasuredItem, *, model_name: str, quant: str,
              batch_size: int, n_samples: int, cache_root: Path):
    cache = GenerationCache(tempfile.mkdtemp(dir=cache_root))
    return sample_item(
        model, cache, model_name=model_name, quant=quant, item_id=item.item_id,
        prompt=item.generation_prompt, batch_size=batch_size, n_samples=n_samples,
        generation_config=f"batch-size-measurement;sample_batch_size={batch_size}",
    )


def _score_prompt(model, item: MeasuredItem) -> None:
    """The main run's per-item prompt-scoring pass (real_run.py), run for its
    memory only. The log-probabilities are discarded unread."""
    score = getattr(model, "score_prompt_detail", None) or model.score_prompt_logprobs
    score(item.item_id, item.scoring_prompt)


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


def generations_sha256(items: list[MeasuredItem], generations_per_item) -> str:
    """SHA-256 over item ids, token ids and log-probabilities of every
    generation, in order. JSON writes floats with repr, which round-trips
    exactly, so a change in any bit of a log-probability changes the digest."""
    digest = hashlib.sha256()
    for item, generations in zip(items, generations_per_item, strict=True):
        for generation in _all_generations(generations):
            digest.update(json.dumps(
                [item.item_id, list(generation.token_ids), list(generation.token_logprobs)]
            ).encode())
    return digest.hexdigest()


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


def _measure_pass(model, items, device, location: list, *, decoding: str, **generate_kwargs):
    device.release()
    device.reset_peak()
    generations_per_item, seconds = [], []
    for item in items:
        location[0] = f"{decoding}:{item.item_id}"
        started = time.perf_counter()
        generations_per_item.append(_generate(model, item, **generate_kwargs))
        _score_prompt(model, item)
        seconds.append(time.perf_counter() - started)
    allocated, reserved = device.peaks()
    measurement = PassMeasurement(
        decoding=decoding, peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
        seconds_per_item=seconds, generated_tokens=summarize_lengths(generations_per_item),
    )
    return measurement, generations_per_item


def measure_candidate(model, items: list[MeasuredItem], batch_size: int, device, *,
                      model_name: str, quant: str, n_samples: int,
                      cache_root: Path) -> CandidateResult:
    result = CandidateResult(batch_size=batch_size, out_of_memory=False)
    generate_kwargs = dict(
        model_name=model_name, quant=quant, batch_size=batch_size,
        n_samples=n_samples, cache_root=cache_root,
    )
    location = [None]
    try:
        result.normal_decoding, first = _measure_pass(
            model, items, device, location, decoding=NORMAL, **generate_kwargs,
        )
        result.generations_sha256 = generations_sha256(items, first)
        location[0] = f"{NORMAL}:repeat"
        repeat = _generate(model, items[0], **generate_kwargs)
        result.reproducible, result.reproducibility_detail = compare_runs(first[0], repeat)
        del first, repeat
        with forced_full_length(model):
            result.forced_length, forced = _measure_pass(
                model, items, device, location, decoding=FORCED, **generate_kwargs,
            )
        result.forced_length_reached = all(
            len(generation.token_ids) == GENERATION_MAX_NEW_TOKENS
            for generations in forced for generation in _all_generations(generations)
        )
        del forced
    except Exception as error:
        if not device.is_out_of_memory(error):
            raise
        result.out_of_memory = True
        result.out_of_memory_at = location[0]
    # Released outside the except block, once the traceback that holds the
    # failed call's tensors is gone.
    device.release()
    return result


def recommend(results: list[CandidateResult], total_memory_bytes: int | None) -> dict:
    rule = (
        f"largest batch size with no out-of-memory, an identical repeat, every forced-length row "
        f"at {GENERATION_MAX_NEW_TOKENS} tokens, and forced-length peak reserved memory <= "
        f"{SAFETY_FRACTION:.2f} x total device memory"
    )
    if total_memory_bytes is None:
        return {"rule": rule, "memory_from": FORCED, "batch_size": None,
                "reason": "device total memory unknown"}
    limit = SAFETY_FRACTION * total_memory_bytes
    acceptable = [
        r.batch_size for r in results
        if not r.out_of_memory and r.reproducible and r.forced_length_reached
        and r.forced_length.peak_reserved_bytes <= limit
    ]
    return {
        "rule": rule,
        "memory_from": FORCED,
        "batch_size": max(acceptable) if acceptable else None,
        "reason": None if acceptable else (
            "no candidate met the rule; rerun with smaller sizes, "
            f"e.g. --batch-sizes {SMALLER_SIZES_HINT}"
        ),
    }


def compare_records(current: dict, previous: dict, previous_path: Path) -> dict:
    """Digest equality per batch size against a record of the same cell
    written by an earlier process."""
    comparison = {"against": str(previous_path)}
    differing = [
        key for key in ("model", "quant", "n_samples", "items")
        if current.get(key) != previous.get(key)
    ]
    if differing:
        return {**comparison, "all_match": False, "per_batch_size": {},
                "reason": f"not the same measurement: {', '.join(differing)} differ"}
    earlier = {
        c["batch_size"]: c["generations_sha256"]
        for c in previous["candidates"] if c.get("generations_sha256")
    }
    per_batch_size = {
        str(c["batch_size"]): c["generations_sha256"] == earlier[c["batch_size"]]
        for c in current["candidates"]
        if c.get("generations_sha256") and c["batch_size"] in earlier
    }
    if not per_batch_size:
        return {**comparison, "all_match": False, "per_batch_size": {},
                "reason": "no batch size has a digest in both records"}
    return {**comparison, "all_match": all(per_batch_size.values()),
            "per_batch_size": per_batch_size, "reason": None}


def exit_code(record: dict) -> int:
    if record["recommendation"]["batch_size"] is None:
        return 1
    if "comparison" in record and not record["comparison"]["all_match"]:
        return 1
    return 0


def _gb(value: int | None) -> str:
    return "-" if value is None else f"{value / 1e9:.1f}"


def print_table(results: list[CandidateResult]) -> None:
    print(f"{'batch':>5} {'pass':<16} {'alloc GB':>8} {'reserved GB':>11} {'s/item':>7} "
          f"{'median tok':>10} {'cap rate':>8}")
    for r in results:
        for measurement in (r.normal_decoding, r.forced_length):
            if measurement is None:
                continue
            print(
                f"{r.batch_size:>5} {measurement.decoding:<16} "
                f"{_gb(measurement.peak_allocated_bytes):>8} {_gb(measurement.peak_reserved_bytes):>11} "
                f"{statistics.fmean(measurement.seconds_per_item):>7.1f} "
                f"{measurement.generated_tokens['median']:>10} "
                f"{measurement.generated_tokens['cap_rate']:>8.2f}"
            )
        if r.out_of_memory:
            print(f"{r.batch_size:>5} out of memory at {r.out_of_memory_at}")
        if r.reproducible is not None:
            print(f"{r.batch_size:>5} repeat {'identical' if r.reproducible else 'DIFFERS'} "
                  f"({r.reproducibility_detail}); generations_sha256 {r.generations_sha256}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="ModelSpec.name from models/registry.py")
    parser.add_argument("--quant", choices=_QUANT_CHOICES, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[50, 25, 10, 5])
    parser.add_argument("--n-items", type=int, default=3,
                        help="How many of the longest LiveCodeBench prompts to measure (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: data/raw/validation/sample_batch_measurement/<model>-<quant>")
    parser.add_argument("--compare-to", type=Path, default=None,
                        help="batch_measurement.json of the same cell from an earlier process; "
                             "exit 1 if any batch size's generations_sha256 differs")
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
    previous = None
    if args.compare_to is not None:
        previous = json.loads(args.compare_to.read_text(encoding="utf-8"))
    spec = get_model(args.model)
    quant = Quant(args.quant)
    batch_sizes = sorted(set(args.batch_sizes), reverse=True)
    output_dir = args.output_dir or _DEFAULT_OUTPUT / f"{spec.name}-{quant.value}"
    device = device or CudaDevice()

    items = load_items()
    model = load_model_fn(spec, quant)
    measured = longest_prompts(items, count_tokens_for(model), args.n_items)
    print(f"Batch-size measurement: model={spec.name}, quant={quant.value}, n_samples={CDD_N_SAMPLES}")
    print("  engineering validation (paper §4.6); records memory, time, generated length and "
          "run-to-run identity only")
    for item in measured:
        print(f"  item {item.item_id}: {item.prompt_tokens} prompt tokens")

    results = []
    with tempfile.TemporaryDirectory(prefix="qcd_batch_measure_") as cache_root:
        for batch_size in batch_sizes:
            result = measure_candidate(
                model, measured, batch_size, device, model_name=spec.name, quant=quant.value,
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
            {"item_id": item.item_id, "prompt_tokens": item.prompt_tokens} for item in measured
        ],
        "candidates": [dataclasses.asdict(result) for result in results],
        "recommendation": recommendation,
    }
    if previous is not None:
        record["comparison"] = compare_records(record, previous, args.compare_to)
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
            "item_ids": [item.item_id for item in measured],
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
    if "comparison" in record:
        comparison = record["comparison"]
        verdict = "all digests match" if comparison["all_match"] else "FAIL"
        reason = f" ({comparison['reason']})" if comparison["reason"] else ""
        print(f"Compared with {comparison['against']}: {verdict}{reason} {comparison['per_batch_size']}")
    print(f"Record: {record_path}")
    return record_path


if __name__ == "__main__":
    sys.exit(exit_code(json.loads(main().read_text())))
