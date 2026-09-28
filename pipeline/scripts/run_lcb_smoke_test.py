#!/usr/bin/env python
"""H100 smoke test for real LiveCodeBench stdin and functional outputs.

Runs one pre- and one post-cutoff item of each test type. This validates the
model-output extraction, public/private test decoding, sandbox scoring, and
parquet round trip before the frozen main run.

`--model` / `--quant` select the arm, the same way `scripts/run_smoke_test.py`
does, so every model in `models/registry.py` and all four precisions of paper
§4.3's ladder can be validated on this path before the main run executes them.
The defaults are unchanged: Qwen2.5-7B-Instruct at BNB-nf4.

**Engineering validation (paper §4.6).** Output goes to the validation-only
namespace `data/raw/validation/lcb_smoke_test/` and its manifest records
`study_phase="engineering_validation"`. Following §4.6, the script checks only
properties of the scoring output — extraction shape, value range, schema — and
does not print or store an item's pass rate or pass@1 outcome. The candidate
code is kept, because the extraction checks are about that text.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import tempfile
import time
from pathlib import Path

import pandas as pd

from qcd.config import Quant
from qcd.constants import LCB_SHARED_CONTROL_BOUNDARY
from qcd.data.livecodebench import load_livecodebench_split
from qcd.generation.cache import GenerationCache
from qcd.generation.sampler import sample_item
from qcd.io.manifest import StudyPhase, build_manifest, write_manifest
from qcd.io.raw_writer import RawDataWriter
from qcd.models.loader import load_model
from qcd.models.registry import QWEN2_5_7B, get_model
from qcd.real_run import _assemble_candidate_code, _generation_prompt
from qcd.scoring.pass_rate import partial_pass_rate
from qcd.scoring.sandbox import _load_test_cases

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Paper §4.6's validation-only namespace, inside pipeline_build_plan.md's
# `data/raw/{validation,main}` split.
_DEFAULT_OUTPUT = _REPO_ROOT / "data" / "raw" / "validation" / "lcb_smoke_test"
_CANDIDATES = {
    ("pre", "stdin"): "1873_A",
    ("pre", "functional"): "2727",
    ("post", "stdin"): "abc387_b",
    ("post", "functional"): "3702",
}


_QUANT_CHOICES = tuple(quant.value for quant in Quant)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        default=QWEN2_5_7B.name,
        help="ModelSpec.name from models/registry.py (default: %(default)s)",
    )
    parser.add_argument("--quant", choices=_QUANT_CHOICES, default=Quant.BNB_NF4.value)
    parser.add_argument("--lcb-release", default="release_v6")
    parser.add_argument("--lcb-cutoff", default=LCB_SHARED_CONTROL_BOUNDARY)
    return parser


def _parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _select_items(cutoff: dt.datetime, release: str):
    pre, post = load_livecodebench_split(cutoff, release_version=release)
    groups = {"pre": {item.item_id: item for item in pre}, "post": {item.item_id: item for item in post}}
    selected = []
    for (split, expected_type), item_id in _CANDIDATES.items():
        item = groups[split][item_id]
        cases = _load_test_cases(item)
        actual_types = {case["testtype"] for case in cases}
        if actual_types != {expected_type}:
            raise RuntimeError(f"{item_id}: expected {expected_type}, found {actual_types}")
        if not item.metadata.get("public_test_cases") or not item.metadata.get("private_test_cases"):
            raise RuntimeError(f"{item_id}: public/private test data is incomplete")
        if expected_type == "functional" and not item.metadata.get("func_name"):
            raise RuntimeError(f"{item_id}: functional item has no func_name")
        selected.append((split, expected_type, item, len(cases)))
    return selected


def main() -> None:
    args = _parse_args()
    model_spec = get_model(args.model)
    quant = Quant(args.quant)
    selected = _select_items(dt.datetime.fromisoformat(args.lcb_cutoff), args.lcb_release)
    print(f"LCB smoke test: model={model_spec.name}, quant={quant.value}")
    model = load_model(model_spec, quant, mock=False)
    cache = GenerationCache(Path(tempfile.mkdtemp(prefix="qcd_lcb_smoke_cache_")))
    writer = RawDataWriter(args.output_dir / "raw")
    writer.write_items([item for _, _, item, _ in selected])
    report = []

    for split, testtype, item, n_tests in selected:
        started = time.perf_counter()
        generations = sample_item(
            model, cache, model_name=model_spec.name, quant=quant.value,
            item_id=item.item_id, prompt=_generation_prompt(item), n_samples=0,
        )
        generation_seconds = time.perf_counter() - started
        candidate = _assemble_candidate_code(item, generations.greedy.text)
        started = time.perf_counter()
        pass_rate = partial_pass_rate(item, candidate)
        sandbox_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prompt_logprobs = model.score_prompt_logprobs(item.item_id, item.prompt)
        prompt_seconds = time.perf_counter() - started

        # `pass_rate` is used for the range check below and nowhere else: §4.6
        # forbids inspecting task-performance outcomes during validation, so it
        # is neither printed, written to the parquet row, nor put in the report.
        checks = {
            "candidate_nonempty": bool(candidate.strip()),
            "markdown_fence_removed": "```" not in candidate,
            "pass_rate_in_range": 0.0 <= pass_rate <= 1.0,
            "prompt_logprobs_present": bool(prompt_logprobs),
            "stdin_program_preserved": testtype != "stdin" or any(
                marker in candidate for marker in ("input(", "sys.stdin", "open(0)")
            ),
            "solution_method_preserved": testtype != "functional" or (
                "class Solution" in candidate and item.metadata["func_name"] in candidate
            ),
        }
        writer.add_generation(
            model=model_spec.name, quant=quant.value,
            item_id=item.item_id, sample_id=0, is_greedy=True,
            text=generations.greedy.text, token_ids=generations.greedy.token_ids,
            token_logprobs=generations.greedy.token_logprobs,
            prompt_token_logprobs=prompt_logprobs,
            decoding_temperature=0.0,
            generation_seconds=generation_seconds,
            prompt_scoring_seconds=prompt_seconds,
            sandbox_scoring_seconds=sandbox_seconds,
        )
        report.append({
            "item_id": item.item_id, "split": split, "testtype": testtype,
            "n_tests": n_tests, "func_name": item.metadata.get("func_name"),
            "generated_text": generations.greedy.text, "candidate_code": candidate,
            "checks": checks,
        })
        print(f"{item.item_id} ({split}/{testtype}): checks={checks}")

    written = writer.flush()
    generations_df = pd.read_parquet(written["generations"])
    parquet_ok = (
        len(generations_df) == len(selected)
        and generations_df["item_id"].notna().all()
        and generations_df["prompt_token_logprobs"].notna().all()
    )
    all_checks = parquet_ok and all(all(row["checks"].values()) for row in report)
    payload = {
        "study_phase": StudyPhase.ENGINEERING_VALIDATION.value,
        "stores_outcome_values": False,
        "all_checks_passed": bool(all_checks),
        "parquet_roundtrip": bool(parquet_ok),
        "items": report,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "lcb_smoke_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest = build_manifest(
        {
            "driver": "scripts/run_lcb_smoke_test.py",
            "model": model_spec.name,
            "model_revision": model_spec.revision,
            "quant": quant.value,
            "lcb_release": args.lcb_release,
            "lcb_cutoff": args.lcb_cutoff,
            "item_ids": [item.item_id for _, _, item, _ in selected],
            "stores_outcome_values": False,
        },
        study_phase=StudyPhase.ENGINEERING_VALIDATION,
        repo_dir=_REPO_ROOT,
    )
    manifest_path = write_manifest(manifest, args.output_dir / "manifest.json")
    print(f"Report: {report_path}")
    print(f"Validation manifest: {manifest_path}")
    if not all_checks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
