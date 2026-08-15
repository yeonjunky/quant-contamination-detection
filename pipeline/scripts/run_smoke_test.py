#!/usr/bin/env python
"""Real nf4 smoke test (Qwen2.5-7B, BNB-nf4) — pipeline/README.md's "Local
smoke-test checklist". A loading/integration check, not a scientific run:
proves the real generate()/score_logprobs() path (models/loader.py) works
end-to-end on real hardware — real quantized load, real sampling, real
sandboxed code execution, real detector scoring, real raw-data writer —
before any pilot-scale run is trusted. Deliberately tiny: 5 shortest-prompt
HumanEval items, 1 greedy + 2 T=0.8 samples each (not the full
CDD_N_SAMPLES=50).

Mirrors qcd/dry_run.py's structure (run everything, print a checklist,
SystemExit(1) on any failed check) but against the real (mock=False) backend.

Usage:
  python scripts/run_smoke_test.py
  python scripts/run_smoke_test.py --model Olmo3-7B-Instruct --quant gptq_awq_int4
  python scripts/run_smoke_test.py --quant gptq_awq_int4 \\
      --checkpoint-path ../data/quantized/Qwen2.5-7B-Instruct-awq-code
    # --checkpoint-path bypasses load_model()'s canonical-path resolution
    # (models/loader.py's _quantized_checkpoint_dir) — lets this script point
    # directly at one of quantize_model.py's calibration-tagged comparison
    # directories (-awq-code/-awq-chat) without copying it to the canonical
    # path first.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from qcd.config import Quant
from qcd.data.humaneval import load_humaneval
from qcd.detectors.cdd import peakedness
from qcd.detectors.mink_prob import mink_prob
from qcd.detectors.perplexity import negative_log_perplexity_score
from qcd.generation.cache import GenerationCache
from qcd.generation.sampler import sample_item
from qcd.io.raw_writer import RawDataWriter
from qcd.models.loader import _RealModelAdapter, load_model
from qcd.models.registry import QWEN2_5_7B, get_model
from qcd.real_run import _assemble_candidate_code
from qcd.scoring.pass_rate import partial_pass_rate

N_ITEMS = 5
N_SAMPLES = 2  # + 1 greedy, per README's checklist (not the full 50)
SAMPLE_TEMPERATURE = 0.8
_QUANT_CHOICES = (Quant.BNB_NF4.value, Quant.GPTQ_AWQ_INT4.value)

# bnb-nf4 keeps weights packed as nf4 through bitsandbytes' own inference
# kernels, so real peak memory tracks the ~4-5GB on-disk size — a tight band
# here catches "accidentally loaded fp16" (~15GB+).
#
# AWQ (llm-compressor/compressed-tensors) does NOT get the same tight band:
# real peak memory measured loading our W4A16_ASYM checkpoints through plain
# `AutoModelForCausalLM.from_pretrained` was ~15-16GB, not ~4-5GB, despite
# the on-disk checkpoint being genuinely ~4-5GB int4 (confirmed 2026-08-15).
# This matches a known compressed-tensors/transformers rough edge with
# asymmetric zero-point decompression (vllm-project/llm-compressor#1550) —
# plain-transformers inference doesn't currently deliver AWQ's memory
# savings the way bnb's dedicated kernels do. The wide band below still
# catches genuine accidents (e.g. an 8x-too-large checkpoint) without
# asserting savings this stack doesn't currently provide.
PLAUSIBLE_PEAK_GB = {
    Quant.BNB_NF4: (2.0, 12.0),
    Quant.GPTQ_AWQ_INT4: (2.0, 20.0),
}

_PIPELINE_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PIPELINE_DIR.parent
# Anchored at the repo root regardless of the invoking CWD, so this always
# lands under the gitignored `/data/` directory (.gitignore's `/data/` is
# root-anchored — a relative "data/..." default only matches that pattern
# when the CWD happens to be the repo root).
_DATA_DIR = _REPO_ROOT / "data" / "smoke_test"


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
    AutoTokenizer, wrapped in the same _RealModelAdapter), so this script can
    point at a quantize_model.py comparison checkpoint that isn't at the
    canonical path yet."""
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
    # other's cached generations (found comparing two AWQ calibration
    # variants: the second run's score_logprobs() failed with "called before
    # generate()" because sample_item() served a cache hit from the first
    # run's checkpoint without ever calling generate() on this run's model).
    quant_label = args.checkpoint_path.name if args.checkpoint_path is not None else quant.value

    items = _select_items(N_ITEMS)
    print(f"Smoke test: {len(items)} HumanEval items, model={model_spec.name}, quant={quant_label}")
    if args.checkpoint_path is not None:
        print(f"  loading from explicit checkpoint path: {args.checkpoint_path}")
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
    # was generated (just in an earlier process) — found comparing two AWQ
    # checkpoints back to back, where the second/third runs both hit this.
    # This script's whole point is exercising the real path every time, not
    # efficiently reusing generations across runs, so skip the cache reuse
    # entirely rather than deepen score_logprobs()'s cross-process contract.
    cache = GenerationCache(Path(tempfile.mkdtemp(prefix="qcd_smoke_cache_")))
    # Tagged by quant_label, not a shared "raw" dir — otherwise a later run
    # (e.g. comparing two --checkpoint-path variants back to back) silently
    # overwrites the previous run's output on disk before it can be compared.
    writer = RawDataWriter(_DATA_DIR / "raw" / quant_label)
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
        pass_rate = partial_pass_rate(item, candidate_code)
        if not (0.0 <= pass_rate <= 1.0):
            pass_rates_ok = False

        # Independent teacher-forced cross-check of score_logprobs() against
        # generate()'s own reported logprobs for the same token sequence.
        tf_scores = model.score_logprobs(item.item_id, generations.greedy.token_ids)
        if len(tf_scores) != len(generations.greedy.token_ids) or not _isfinite_all(tf_scores):
            teacher_forced_scoring_ok = False

        cdd_score = peakedness(generations.greedy.token_ids, [s.token_ids for s in generations.samples])
        ppl_score = negative_log_perplexity_score(generations.greedy.token_logprobs)
        mink_score = mink_prob(generations.greedy.token_logprobs)
        if not (0.0 <= cdd_score <= 1.0) or not math.isfinite(ppl_score) or not math.isfinite(mink_score):
            detector_scores_ok = False

        print(f"    pass_rate={pass_rate:.2f} cdd={cdd_score:.2f} ppl={ppl_score:.3f} mink={mink_score:.3f}")

        writer.add_generation(
            model=model_spec.name, quant=quant_label, item_id=item.item_id, sample_id=0, is_greedy=True,
            text=generations.greedy.text, token_ids=generations.greedy.token_ids,
            token_logprobs=generations.greedy.token_logprobs, partial_pass_rate=pass_rate, decoding_temperature=0.0,
        )
        for sample_id, sample in enumerate(generations.samples, start=1):
            writer.add_generation(
                model=model_spec.name, quant=quant_label, item_id=item.item_id, sample_id=sample_id,
                is_greedy=False, text=sample.text, token_ids=sample.token_ids,
                token_logprobs=sample.token_logprobs, decoding_temperature=SAMPLE_TEMPERATURE,
            )
        for detector, score in (("cdd", cdd_score), ("perplexity", ppl_score), ("mink_prob", mink_score)):
            writer.add_detector_score(model=model_spec.name, quant=quant_label, item_id=item.item_id, detector=detector, score=score)

    written = writer.flush()
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nPeak GPU memory: {peak_gb:.2f} GB")

    freeze_path = _save_pip_freeze()

    checks = {
        "logprobs_finite": all_finite,
        "repeated_samples_differ": samples_differ,
        "sandbox_pass_rate_in_range": pass_rates_ok,
        "teacher_forced_scoring_ok": teacher_forced_scoring_ok,
        "detector_scores_plausible": detector_scores_ok,
        "peak_memory_in_band": PLAUSIBLE_PEAK_GB[quant][0] <= peak_gb <= PLAUSIBLE_PEAK_GB[quant][1],
        "writer_output_matches_mock_schema": (
            (_DATA_DIR / "raw" / quant_label / "items.parquet").exists() and "generations" in written and "detector_scores" in written
        ),
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


if __name__ == "__main__":
    main()
