#!/usr/bin/env python
"""Real-hardware smoke test — pipeline/README.md's "Local smoke-test
checklist". A loading/integration check, not a scientific run: proves the real
generate()/score_logprobs() path (models/loader.py) works end-to-end on real
hardware — real load at the requested precision, real sampling, real sandboxed
code execution, real detector scoring, real raw-data writer — before the frozen
main run begins. Deliberately tiny: 5 shortest-prompt HumanEval items,
1 greedy + 2 T=0.8 samples each (not the full CDD_N_SAMPLES=50).

**This is engineering validation (paper §4.6).** Output goes to the
validation-only namespace `data/raw/validation/smoke_test/` and its manifest
records `study_phase="engineering_validation"`, so the analysis side refuses
it. §4.6 also forbids letting outcome values steer the configuration, so this
script checks only *properties* of the numbers — finite, in range, right
schema — and neither prints nor stores an item's pass rate or detector scores.
Both quantities are still computed, because the range checks are the point.

Mirrors qcd/dry_run.py's structure (run everything, print a checklist,
SystemExit(1) on any failed check) but against the real (mock=False) backend.

Usage:
  python scripts/run_smoke_test.py
  python scripts/run_smoke_test.py --model Qwen2.5-32B-Instruct --quant bnb_int8
  python scripts/run_smoke_test.py --quant gptq_awq_int4 \\
      --checkpoint-path ../data/quantized/Qwen2.5-7B-Instruct-awq
    # --checkpoint-path bypasses load_model()'s canonical-path resolution
    # (models/loader.py's _quantized_checkpoint_dir) to load a checkpoint
    # sitting somewhere else on disk.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from qcd.config import ModelSpec, Quant
from qcd.data.humaneval import load_humaneval
from qcd.detectors.cdd import peakedness
from qcd.detectors.mink_prob import mink_prob
from qcd.detectors.perplexity import negative_log_perplexity_score
from qcd.generation.cache import GenerationCache
from qcd.generation.sampler import sample_item
from qcd.io.manifest import (
    StudyPhase, build_manifest, resolve_library_defaults,
    unresolved_library_defaults, write_manifest,
)
from qcd.io.raw_writer import RawDataWriter
from qcd.models.loader import _RealModelAdapter, load_model
from qcd.models.registry import QWEN2_5_7B, get_model
from qcd.real_run import _assemble_candidate_code
from qcd.scoring.pass_rate import partial_pass_rate

N_ITEMS = 5
N_SAMPLES = 2  # + 1 greedy, per README's checklist (not the full 50)
SAMPLE_TEMPERATURE = 0.8
_QUANT_CHOICES = tuple(quant.value for quant in Quant)

# Memory band, per model *and* precision. A single band per precision could not
# hold both size classes: paper §4.1's footprint table puts a 7-8B nf4 load at
# ~4-5 GB and a 32B nf4 load at ~18 GB, so one 12 GB ceiling would fail every
# 32B condition. The band is derived from the weight footprint instead of typed
# per condition — it exists to catch "silently loaded at the wrong precision"
# (an 8x error), not to certify a savings figure.
_BYTES_PER_PARAMETER = {
    Quant.BF16: 2.0,
    Quant.BNB_INT8: 1.0,
    Quant.BNB_NF4: 0.5,
    Quant.GPTQ_AWQ_INT4: 0.5,
}
_LOWER_FACTOR = 0.6  # below this, the weights cannot be at the requested precision
_UPPER_FACTOR = 2.0
_OVERHEAD_GB = 4.0  # KV cache, activations, allocator slack


def plausible_peak_gb(spec: ModelSpec, quant: Quant) -> tuple[float, float]:
    """(lower, upper) GB band for peak allocated GPU memory.

    AWQ gets a bf16-width ceiling on purpose: real peak memory measured loading
    our W4A16_ASYM checkpoints through plain `AutoModelForCausalLM.from_pretrained`
    was ~15-16 GB for a 7B model, not ~4-5 GB, despite the on-disk checkpoint
    being genuinely ~4-5 GB int4 (confirmed 2026-08-15). That matches a known
    compressed-tensors/transformers rough edge with asymmetric zero-point
    decompression (vllm-project/llm-compressor#1550) — plain-transformers
    inference does not currently deliver AWQ's memory savings the way bnb's
    dedicated kernels do. The lower bound still catches an implausibly small
    load; the upper bound does not assert savings this stack does not provide.
    """
    weight_gb = spec.param_count_b * _BYTES_PER_PARAMETER[quant]
    ceiling_basis = (
        spec.param_count_b * _BYTES_PER_PARAMETER[Quant.BF16]
        if quant is Quant.GPTQ_AWQ_INT4
        else weight_gb
    )
    return _LOWER_FACTOR * weight_gb, _UPPER_FACTOR * ceiling_basis + _OVERHEAD_GB


_PIPELINE_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PIPELINE_DIR.parent
# Paper §4.6's validation-only namespace, anchored at the repo root regardless
# of the invoking CWD so it always lands under the gitignored `/data/`
# (.gitignore's `/data/` is root-anchored) and inside the
# `data/raw/{validation,main}` split pipeline_build_plan.md describes.
_DATA_DIR = _REPO_ROOT / "data" / "raw" / "validation" / "smoke_test"


def _isfinite_all(values) -> bool:
    return all(math.isfinite(v) for v in values)


def _select_items(n: int):
    items = load_humaneval()
    return sorted(items, key=lambda item: len(item.prompt))[:n]


def _save_pip_freeze() -> Path:
    envs_dir = _PIPELINE_DIR / "envs"
    envs_dir.mkdir(parents=True, exist_ok=True)
    out_path = envs_dir / "local-smoke-freeze.txt"
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    out_path.write_text(freeze)
    return out_path


def _load_model_for_smoke_test(spec, quant: Quant, checkpoint_path: Path | None):
    """`checkpoint_path`, when given, bypasses load_model()'s canonical-path
    resolution entirely — loads straight from that directory the same way
    models/loader.py's real backends do (plain AutoModelForCausalLM +
    AutoTokenizer, wrapped in the same _RealModelAdapter)."""
    if checkpoint_path is None:
        return load_model(spec, quant, mock=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map="auto")
    return _RealModelAdapter(model, tokenizer)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=QWEN2_5_7B.name, help="ModelSpec.name from models/registry.py (default: %(default)s)")
    parser.add_argument("--quant", choices=_QUANT_CHOICES, default=Quant.BNB_NF4.value)
    parser.add_argument(
        "--checkpoint-path", type=Path, default=None,
        help="Load directly from this local checkpoint dir instead of load_model()'s canonical-path "
             "resolution (see module docstring's --checkpoint-path example).",
    )
    return parser.parse_args()


def main() -> None:
    import torch  # noqa: PLC0415

    args = _parse_args()
    model_spec = get_model(args.model)
    quant = Quant(args.quant)
    # Distinguishes cache entries/written rows by checkpoint, not just Quant
    # level — without this, two different --checkpoint-path runs sharing the
    # same (model, quant) collide in GenerationCache and silently serve each
    # other's cached generations (found comparing two AWQ checkpoints: the
    # second run's score_logprobs() failed with "called before generate()"
    # because sample_item() served a cache hit from the first run's checkpoint
    # without ever calling generate() on this run's model).
    quant_label = args.checkpoint_path.name if args.checkpoint_path is not None else quant.value

    items = _select_items(N_ITEMS)
    print(f"Smoke test: {len(items)} HumanEval items, model={model_spec.name}, quant={quant_label}")
    if args.checkpoint_path is not None:
        print(f"  loading from explicit checkpoint path: {args.checkpoint_path}")
    print(f"  engineering validation (paper §4.6) — output namespace: {_DATA_DIR}")
    print(
        "  pass rates and detector scores are computed for the range checks below and are "
        "neither printed nor stored (§4.6)."
    )
    print()

    torch.cuda.reset_peak_memory_stats()
    t_load = time.time()
    model = _load_model_for_smoke_test(model_spec, quant, args.checkpoint_path)
    print(f"Model load: {time.time() - t_load:.1f}s")

    # A fresh temp dir every invocation, not a persistent _DATA_DIR/cache —
    # score_logprobs() tracks its prompt in-memory per adapter instance,
    # keyed off generate() having run first (models/loader.py). A cache hit
    # across separate script invocations skips generate() on the *this run's*
    # freshly-loaded model, so score_logprobs()'s teacher-forced cross-check
    # then fails with "called before generate()" even though the item really
    # was generated (just in an earlier process). This script's whole point is
    # exercising the real path every time, not efficiently reusing generations
    # across runs, so skip the cache reuse entirely rather than deepen
    # score_logprobs()'s cross-process contract.
    cache = GenerationCache(Path(tempfile.mkdtemp(prefix="qcd_smoke_cache_")))
    # Tagged by quant_label, not a shared "raw" dir — otherwise a later run
    # silently overwrites the previous run's output on disk.
    run_dir = _DATA_DIR / quant_label
    writer = RawDataWriter(run_dir / "raw", file_prefix=model_spec.name)
    writer.write_items(items)

    all_finite = True
    samples_differ = True
    pass_rates_ok = True
    detector_scores_ok = True
    teacher_forced_scoring_ok = True

    for item in items:
        t0 = time.time()
        generations = sample_item(
            model, cache, model_name=model_spec.name, quant=quant_label,
            item_id=item.item_id, prompt=item.prompt, n_samples=N_SAMPLES, sample_temperature=SAMPLE_TEMPERATURE,
        )
        print(f"  {item.item_id}: {time.time() - t0:.1f}s, greedy {len(generations.greedy.token_ids)} tokens")

        for gen in [generations.greedy, *generations.samples]:
            if not gen.token_logprobs or not _isfinite_all(gen.token_logprobs):
                all_finite = False

        if len(generations.samples) >= 2 and generations.samples[0].token_ids == generations.samples[1].token_ids:
            samples_differ = False

        candidate_code = _assemble_candidate_code(item, generations.greedy.text)
        # Range check only: the value is deliberately not printed, not written
        # to the parquet rows, and not compared across items or precisions.
        pass_rate = partial_pass_rate(item, candidate_code)
        if not (0.0 <= pass_rate <= 1.0):
            pass_rates_ok = False

        # Keep the completion-confidence cross-check, and separately exercise
        # the paper's fixed-prompt detector path. The latter must work without
        # relying on generate()'s in-memory prompt history.
        tf_scores = model.score_logprobs(item.item_id, generations.greedy.token_ids)
        if len(tf_scores) != len(generations.greedy.token_ids) or not _isfinite_all(tf_scores):
            teacher_forced_scoring_ok = False
        prompt_logprobs = model.score_prompt_logprobs(item.item_id, item.prompt)
        if not prompt_logprobs or not _isfinite_all(prompt_logprobs):
            teacher_forced_scoring_ok = False

        # Same rule as pass_rate: every detector is exercised so its code path
        # and output range are validated, and no score leaves this loop.
        cdd_score = peakedness(generations.greedy.token_ids, [s.token_ids for s in generations.samples])
        ppl_score = negative_log_perplexity_score(prompt_logprobs)
        mink_score = mink_prob(prompt_logprobs)
        completion_ppl_score = negative_log_perplexity_score(generations.greedy.token_logprobs)
        completion_mink_score = mink_prob(generations.greedy.token_logprobs)
        if not all((
            0.0 <= cdd_score <= 1.0,
            math.isfinite(ppl_score), math.isfinite(mink_score),
            math.isfinite(completion_ppl_score), math.isfinite(completion_mink_score),
        )):
            detector_scores_ok = False

        writer.add_generation(
            model=model_spec.name, quant=quant_label, item_id=item.item_id, sample_id=0, is_greedy=True,
            text=generations.greedy.text, token_ids=generations.greedy.token_ids,
            token_logprobs=generations.greedy.token_logprobs,
            prompt_token_logprobs=prompt_logprobs,
            decoding_temperature=0.0,
        )
        for sample_id, sample in enumerate(generations.samples, start=1):
            writer.add_generation(
                model=model_spec.name, quant=quant_label, item_id=item.item_id, sample_id=sample_id,
                is_greedy=False, text=sample.text, token_ids=sample.token_ids,
                token_logprobs=sample.token_logprobs, decoding_temperature=SAMPLE_TEMPERATURE,
            )

    written = writer.flush()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    lower_gb, upper_gb = plausible_peak_gb(model_spec, quant)
    print(f"\nPeak GPU memory: {peak_gb:.2f} GB (expected band {lower_gb:.1f}-{upper_gb:.1f} GB)")

    freeze_path = _save_pip_freeze()
    manifest_path = _write_validation_manifest(
        run_dir, model_spec=model_spec, quant=quant, quant_label=quant_label,
        model=model, n_items=len(items),
    )
    print(f"Validation manifest: {manifest_path}")

    checks = {
        "logprobs_finite": all_finite,
        "repeated_samples_differ": samples_differ,
        "sandbox_pass_rate_in_range": pass_rates_ok,
        "teacher_forced_scoring_ok": teacher_forced_scoring_ok,
        "detector_scores_in_range": detector_scores_ok,
        "peak_memory_in_band": lower_gb <= peak_gb <= upper_gb,
        "writer_output_matches_mock_schema": (
            (run_dir / "raw" / f"{model_spec.name}_items.parquet").exists()
            and "generations" in written
        ),
        "validation_manifest_written": manifest_path.exists(),
        "pip_freeze_saved": freeze_path.exists(),
    }

    print("\nChecklist:")
    for name, ok in checks.items():
        print(f"  [{'x' if ok else ' '}] {name}")

    failures = [name for name, ok in checks.items() if not ok]
    if failures:
        print(f"\nFAILED: {failures}")
        raise SystemExit(1)
    print("\nAll smoke-test checks passed.")


def _write_validation_manifest(
    run_dir: Path, *, model_spec, quant: Quant, quant_label: str, model, n_items: int
) -> Path:
    """Paper §4.6: validation output is recorded as validation output.

    Also carries §4.3's resolved library defaults for the precision actually
    loaded — this is where the BNB skip list and block size can be read from a
    real quantized model before the main run.
    """
    library_defaults = resolve_library_defaults(model=getattr(model, "model", model))
    manifest = build_manifest(
        {
            "driver": "scripts/run_smoke_test.py",
            "model": model_spec.name,
            "model_revision": model_spec.revision,
            "quant": quant.value,
            "quant_label": quant_label,
            "n_items": n_items,
            "n_samples": N_SAMPLES,
            "sample_temperature": SAMPLE_TEMPERATURE,
            "dataset": "humaneval",
            "stores_outcome_values": False,
        },
        study_phase=StudyPhase.ENGINEERING_VALIDATION,
        repo_dir=_REPO_ROOT,
        extra={
            "library_default_settings": library_defaults,
            "library_default_settings_unresolved": unresolved_library_defaults(library_defaults),
        },
    )
    return write_manifest(manifest, run_dir / "manifest.json")


if __name__ == "__main__":
    main()
