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

Usage: python scripts/run_smoke_test.py
"""

from __future__ import annotations

import math
import subprocess
import sys
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
from qcd.models.loader import load_model
from qcd.models.registry import QWEN2_5_7B
from qcd.real_run import _assemble_candidate_code
from qcd.scoring.pass_rate import partial_pass_rate

N_ITEMS = 5
N_SAMPLES = 2  # + 1 greedy, per README's checklist (not the full 50)
SAMPLE_TEMPERATURE = 0.8
QUANT = Quant.BNB_NF4

# Generous band around the ~4-5GB nf4-quantized Qwen2.5-7B is expected to
# occupy — wide enough to tolerate activation/KV-cache overhead, tight enough
# to catch "accidentally loaded fp16" (which would be ~15GB+).
PLAUSIBLE_NF4_PEAK_GB = (2.0, 12.0)

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


def main() -> None:
    import torch  # noqa: PLC0415

    items = _select_items(N_ITEMS)
    print(f"Smoke test: {len(items)} HumanEval items, model={QWEN2_5_7B.name}, quant={QUANT.value}\n")

    torch.cuda.reset_peak_memory_stats()
    t_load = time.time()
    model = load_model(QWEN2_5_7B, QUANT, mock=False)
    print(f"Model load: {time.time() - t_load:.1f}s")

    cache = GenerationCache(_DATA_DIR / "cache")
    writer = RawDataWriter(_DATA_DIR / "raw")
    writer.write_items(items)

    all_finite = True
    samples_differ = True
    pass_rates_ok = True
    detector_scores_ok = True
    teacher_forced_scoring_ok = True

    for item in items:
        t0 = time.time()
        generations = sample_item(
            model, cache, model_name=QWEN2_5_7B.name, quant=QUANT.value,
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

        writer.add_generation(
            model=QWEN2_5_7B.name, quant=QUANT.value, item_id=item.item_id, sample_id=0, is_greedy=True,
            text=generations.greedy.text, token_ids=generations.greedy.token_ids,
            token_logprobs=generations.greedy.token_logprobs, partial_pass_rate=pass_rate, decoding_temperature=0.0,
        )
        for sample_id, sample in enumerate(generations.samples, start=1):
            writer.add_generation(
                model=QWEN2_5_7B.name, quant=QUANT.value, item_id=item.item_id, sample_id=sample_id,
                is_greedy=False, text=sample.text, token_ids=sample.token_ids,
                token_logprobs=sample.token_logprobs, decoding_temperature=SAMPLE_TEMPERATURE,
            )
        for detector, score in (("cdd", cdd_score), ("perplexity", ppl_score), ("mink_prob", mink_score)):
            writer.add_detector_score(model=QWEN2_5_7B.name, quant=QUANT.value, item_id=item.item_id, detector=detector, score=score)

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
        "nf4_peak_memory_in_band": PLAUSIBLE_NF4_PEAK_GB[0] <= peak_gb <= PLAUSIBLE_NF4_PEAK_GB[1],
        "writer_output_matches_mock_schema": (
            (_DATA_DIR / "raw" / "items.parquet").exists() and "generations" in written and "detector_scores" in written
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
