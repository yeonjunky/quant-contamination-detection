#!/usr/bin/env python
"""One-time, offline AWQ quantization (llm-compressor) for one model in
`models/registry.py`. Produces the canonical local checkpoint
`models/loader.py`'s `_load_gptq_or_awq` loads with plain
`AutoModelForCausalLM.from_pretrained` — quantization itself is a deliberate,
separate step, never implicit inside a real experiment run.

Uses llm-compressor uniformly for all five roster models (not GPTQModel):
GPTQModel's own architecture registry (gptqmodel/models/auto.py) has no
`olmo3` entry (`olmo2` maps to `LlamaQModel`; `olmo`/`olmo3` are absent, as
of this session's check against its source), so it would very likely fail
on the two Olmo3 arms. llm-compressor has no per-architecture registry — its
`AWQModifier`/`QuantizationModifier` recipe targets any HF-loadable causal
LM's `nn.Linear` layers by name pattern, so it's expected to work on Olmo3
without needing upstream support. The paper (`paper/paper_draft.md` §4.3)
specifies AWQ uniformly, so the implementation does not mix calibration-based
quantizers across models.

Recipe verbatim from vllm-project/llm-compressor's own reference example
(examples/awq/llama_example.py) — the canonical llama-family recipe, which
Qwen2.5/Llama-3.1/Olmo3 all fit (dense, standard nn.Linear projections).

**Calibration data is fixed, not selected.** Paper §4.3: "Before the main run,
freeze one AWQ calibration artifact per model ... Exactly one calibration
artifact per model enters the main run." Every model is quantized with the
same **code** calibration set — `flytech/python-codes-25k`, the pinned
revision below, 256 rows shuffled with seed 42, max sequence length 512, read
from the dataset's own `text` column as shipped with no chat template. No
calibration variant is compared against another, and none is selected
afterwards: this script writes `data/quantized/<model>-awq/` — the canonical
path `models/loader.py` reads — directly. The chat set named in §4.3 is an
engineering-comparison artifact from earlier work; it does not enter the main
run and this script does not produce it.

That dataset went through two rejections before landing:
`bigcode/the-stack-smol` (the original choice) turned out to be gated
(`DatasetNotFoundError`, discovered empirically running this script), and
`codeparrot/github-code-clean` uses a legacy dataset-loading script
`datasets>=5` no longer supports (the same failure mode
`data/livecodebench.py`'s docstring already documents for LiveCodeBench).
`flytech/python-codes-25k` is ungated, loads cleanly, and its `text` field is
already instruction + brief explanation + a fenced Python code block.

Alongside the checkpoint this script writes two records paper §4.3 requires:

  - `quantization_manifest.json` — dataset revision, the seed, the sha256 of
    every selected calibration row and of the selected-row list as a whole,
    the tokenizer, the recipe, the resolved AWQ group size, and the installed
    software versions.
  - `calibration_overlap_report.json` — the overlap check of the calibration
    text against the evaluation prompts (LiveCodeBench `question_content`,
    HumanEval+ and MBPP+ prompts, and the two evalplus arms' reference
    solutions). `models/loader.py` refuses to load a checkpoint without it.

Usage: python scripts/quantize_model.py <model-name>
  e.g. python scripts/quantize_model.py Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path

from qcd.constants import LCB_SHARED_CONTROL_BOUNDARY
from qcd.io.manifest import (
    AWQ_PRESET_SCHEME, CALIBRATION_OVERLAP_NGRAM_SIZE, StudyPhase,
    build_manifest, calibration_overlap_report, resolve_awq_group_size,
    write_calibration_overlap_report, write_manifest,
)
from qcd.models.registry import get_model

_PIPELINE_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PIPELINE_DIR.parent
_QUANTIZED_DIR = _REPO_ROOT / "data" / "quantized"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512
#: Paper §4.3's "256 rows shuffled with seed 42" — recorded in the manifest
#: rather than only living in a call argument.
CALIBRATION_SHUFFLE_SEED = 42
CALIBRATION_TEXT_COLUMN = "text"

CALIBRATION_DATASET_ID = "flytech/python-codes-25k"
CALIBRATION_DATASET_REVISION = "0ed98ff2a76c5d133d8c157b814189a5a17ebd20"
CALIBRATION_DATASET_SPLIT = "train"


def _checkpoint_dir(model_name: str) -> Path:
    """The canonical path `models/loader.py::_quantized_checkpoint_dir` reads."""
    return _QUANTIZED_DIR / f"{model_name}-awq"


def _load_calibration_dataset():
    from datasets import load_dataset  # noqa: PLC0415

    dataset = load_dataset(
        CALIBRATION_DATASET_ID, revision=CALIBRATION_DATASET_REVISION,
        split=CALIBRATION_DATASET_SPLIT,
    )
    return dataset.shuffle(seed=CALIBRATION_SHUFFLE_SEED).select(range(NUM_CALIBRATION_SAMPLES))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def calibration_row_hashes(texts: list[str]) -> dict:
    """Per-row sha256 plus one hash over the ordered list — §4.3's
    "selected-row hashes". The list hash makes "is this the same selection?"
    a single comparison; the per-row hashes say which row differs when it is
    not."""
    row_hashes = [_sha256(text) for text in texts]
    return {
        "row_sha256": row_hashes,
        "selected_rows_sha256": _sha256("\n".join(row_hashes)),
    }


def evaluation_texts(lcb_release: str = "release_v6"):
    """Every evaluation text §4.3 asks the calibration set to be checked
    against: the benchmark prompts, plus the two evalplus arms' reference
    solutions (LiveCodeBench ships no public reference solutions)."""
    from qcd.data.humaneval import load_humaneval  # noqa: PLC0415
    from qcd.data.livecodebench import load_livecodebench_split  # noqa: PLC0415
    from qcd.data.mbppplus import load_mbppplus  # noqa: PLC0415

    boundary = dt.datetime.fromisoformat(LCB_SHARED_CONTROL_BOUNDARY)
    pre, post = load_livecodebench_split(boundary, release_version=lcb_release)
    for item in list(pre) + list(post):
        yield ("livecodebench_question_content", item.item_id, item.prompt)

    for loader, name in ((load_humaneval, "humaneval"), (load_mbppplus, "mbppplus")):
        for item in loader():
            yield (f"{name}_prompt", item.item_id, item.prompt)
            solution = item.metadata.get("evalplus_problem", {}).get("canonical_solution")
            if solution:
                yield (f"{name}_canonical_solution", item.item_id, solution)


def quantize(model_name: str, *, lcb_release: str = "release_v6") -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    from llmcompressor import oneshot  # noqa: PLC0415
    from llmcompressor.modifiers.quantization import QuantizationModifier  # noqa: PLC0415
    from llmcompressor.modifiers.transform.awq import AWQModifier  # noqa: PLC0415

    spec = get_model(model_name)
    save_dir = _checkpoint_dir(spec.name)

    print(f"Loading {spec.hf_repo_id} (bf16 source) for quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_repo_id, revision=spec.revision, torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo_id, revision=spec.revision)

    print(f"Loading code calibration dataset ({NUM_CALIBRATION_SAMPLES} rows, seed {CALIBRATION_SHUFFLE_SEED})...")
    dataset = _load_calibration_dataset()
    calibration_texts = [str(row) for row in dataset[CALIBRATION_TEXT_COLUMN]]
    hashes = calibration_row_hashes(calibration_texts)

    calibration_record = {
        "dataset_id": CALIBRATION_DATASET_ID,
        "dataset_revision": CALIBRATION_DATASET_REVISION,
        "dataset_split": CALIBRATION_DATASET_SPLIT,
        "text_column": CALIBRATION_TEXT_COLUMN,
        "chat_template_applied": False,
        "shuffle_seed": CALIBRATION_SHUFFLE_SEED,
        "num_calibration_samples": NUM_CALIBRATION_SAMPLES,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "tokenizer": spec.hf_repo_id,
        "tokenizer_revision": spec.revision,
        **hashes,
    }

    # Run the overlap check BEFORE the expensive oneshot pass: §4.3 calls it a
    # pre-execution step, and a failure here should not cost the calibration
    # run.
    print("Checking calibration text against the evaluation prompts...")
    overlap = calibration_overlap_report(
        calibration_texts=calibration_texts,
        evaluation_texts=evaluation_texts(lcb_release),
        ngram_size=CALIBRATION_OVERLAP_NGRAM_SIZE,
        calibration=calibration_record,
    )
    print(
        f"  {overlap['n_overlapping_texts']} of {overlap['n_evaluation_texts']} evaluation texts "
        f"share a {CALIBRATION_OVERLAP_NGRAM_SIZE}-token n-gram with the calibration set"
    )

    recipe = [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(ignore=["lm_head"], scheme=AWQ_PRESET_SCHEME, targets=["Linear"]),
    ]

    print("Running AWQ oneshot calibration (this is the expensive step)...")
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    print("\n========== SAMPLE GENERATION (sanity check) ==============")
    input_ids = tokenizer("def add(a, b):\n    ", return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=60)
    print(tokenizer.decode(output[0]))
    print("=============================================================\n")

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved quantized checkpoint to {save_dir}")

    report_path = write_calibration_overlap_report(overlap, save_dir)
    print(f"Wrote overlap report to {report_path}")

    manifest = build_manifest(
        {
            "model_name": spec.name,
            "hf_repo_id": spec.hf_repo_id,
            "source_model_revision": spec.revision,
            "quant": "gptq_awq_int4",
            "backend": "llmcompressor_awq",
            "calibration_domain": "code",
            "calibration": calibration_record,
            "recipe": [
                "AWQModifier(duo_scaling=both)",
                f"QuantizationModifier(scheme={AWQ_PRESET_SCHEME}, targets=Linear, ignore=lm_head)",
            ],
            # §4.3: "group size left at the scheme's default" — recorded as the
            # value it resolved to, read from the quantized model itself where
            # possible and from the preset otherwise.
            "resolved_awq_group_size": resolve_awq_group_size(model),
            "calibration_overlap_check": {
                "n_evaluation_texts": overlap["n_evaluation_texts"],
                "n_overlapping_texts": overlap["n_overlapping_texts"],
                "report": report_path.name,
            },
        },
        # The frozen calibration artifact is an input to the main run
        # (§4.3: "Exactly one calibration artifact per model enters the main
        # run"), not engineering-validation output.
        study_phase=StudyPhase.MAIN_STUDY,
        seed=CALIBRATION_SHUFFLE_SEED,
        repo_dir=_REPO_ROOT,
    )
    write_manifest(manifest, save_dir / "quantization_manifest.json")

    return save_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_name", help="ModelSpec.name, e.g. Qwen2.5-7B-Instruct (see models/registry.py)")
    parser.add_argument(
        "--lcb-release", default="release_v6",
        help="LiveCodeBench release whose question_content is checked for overlap (default: %(default)s)",
    )
    args = parser.parse_args()

    save_dir = quantize(args.model_name, lcb_release=args.lcb_release)
    print(f"Done. Checkpoint written to the canonical loader.py path: {save_dir}")


if __name__ == "__main__":
    main()
