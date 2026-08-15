#!/usr/bin/env python
"""One-time, offline AWQ quantization (llm-compressor) for one model in
`models/registry.py`. Produces a local checkpoint `models/loader.py`'s
`_load_gptq_or_awq` can load with plain `AutoModelForCausalLM.from_pretrained`
— quantization itself is a deliberate, separate step, never implicit inside
a real experiment run.

Uses llm-compressor uniformly for all five roster models (not GPTQModel):
GPTQModel's own architecture registry (gptqmodel/models/auto.py) has no
`olmo3` entry (`olmo2` maps to `LlamaQModel`; `olmo`/`olmo3` are absent, as
of this session's check against its source), so it would very likely fail
on the two Olmo3 arms. llm-compressor has no per-architecture registry — its
`AWQModifier`/`QuantizationModifier` recipe targets any HF-loadable causal
LM's `nn.Linear` layers by name pattern, so it's expected to work on Olmo3
without needing upstream support. The paper (`paper/paper_draft.md` §4.3)
treats "GPTQ-int4 or AWQ-int4" as interchangeable representatives of one
calibration-based 4-bit condition, not a per-model design axis, so using one
technique uniformly is more consistent than mixing, not less.

Recipe verbatim from vllm-project/llm-compressor's own reference example
(examples/awq/llama_example.py) — the canonical llama-family recipe, which
Qwen2.5/Llama-3.1/Olmo3 all fit (dense, standard nn.Linear projections).

**Calibration data — this is deliberately a live comparison, not a fixed
choice** (see pipeline_build_plan.md / pipeline_implementation_log.md's
2026-08-15 entry): the paper's own literature review (§2.7/§4.3) argues
calibration-based methods are more forgiving specifically on code-
specialized models, implying calibration *domain* matters — but that's a
hypothesis. `--calibration code` tests that hypothesis; `--calibration chat`
(HuggingFaceH4/ultrachat_200k, chat-templated) is llm-compressor's own
reference default, used here as the comparison baseline. Both are saved
under a calibration-tagged directory name — neither is the canonical path
`models/loader.py` reads (`data/quantized/<model>-awq/`, no suffix); after
comparing the two (scripts/run_smoke_test.py --checkpoint-path), copy or
re-quantize the winner to the canonical path by hand.

`--calibration code`'s dataset went through two rejections before landing:
`bigcode/the-stack-smol` (the original choice) turned out to be gated
(`DatasetNotFoundError`, discovered empirically running this script), and
`codeparrot/github-code-clean` uses a legacy dataset-loading script
`datasets>=5` no longer supports (the same failure mode
`data/livecodebench.py`'s docstring already documents for LiveCodeBench).
Landed on `flytech/python-codes-25k` — ungated, loads cleanly, and its
`text` field is already instruction + brief explanation + a fenced Python
code block, arguably *more* representative of this pipeline's actual
inference-time distribution (an -Instruct model answering a code-generation
prompt) than raw unstructured source files would have been.

Usage: python scripts/quantize_model.py <model-name> --calibration {code,chat}
  e.g. python scripts/quantize_model.py Qwen2.5-7B-Instruct --calibration code
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qcd.io.manifest import build_manifest, write_manifest
from qcd.models.registry import get_model

_PIPELINE_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PIPELINE_DIR.parent
_QUANTIZED_DIR = _REPO_ROOT / "data" / "quantized"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

_CODE_DATASET_ID = "flytech/python-codes-25k"
_CHAT_DATASET_ID = "HuggingFaceH4/ultrachat_200k"


def _checkpoint_dir(model_name: str, calibration: str) -> Path:
    return _QUANTIZED_DIR / f"{model_name}-awq-{calibration}"


def _load_code_calibration_dataset(tokenizer):
    from datasets import load_dataset  # noqa: PLC0415

    # bigcode/the-stack-smol (the original choice) is a gated dataset —
    # discovered empirically (DatasetNotFoundError requiring authentication)
    # while first running this script. codeparrot/github-code-clean, the
    # next candidate, uses a legacy dataset-loading script that datasets>=5
    # no longer supports (same "Dataset scripts are no longer supported"
    # error data/livecodebench.py's docstring already documents hitting for
    # LiveCodeBench). flytech/python-codes-25k is ungated, loads cleanly
    # under the installed datasets version, and its `text` field is already
    # instruction + brief explanation + a fenced Python code block — which,
    # unlike raw unstructured source files, is arguably *more* representative
    # of this pipeline's actual inference-time distribution (an -Instruct
    # model answering a code-generation prompt), not less.
    ds = load_dataset(_CODE_DATASET_ID, split="train")
    return ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


def _load_chat_calibration_dataset(tokenizer):
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(_CHAT_DATASET_ID, split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
    ds = ds.shuffle(seed=42)

    def _to_text(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    return ds.map(_to_text)


def _load_calibration_dataset(calibration: str, tokenizer):
    if calibration == "code":
        return _load_code_calibration_dataset(tokenizer)
    if calibration == "chat":
        return _load_chat_calibration_dataset(tokenizer)
    raise ValueError(f"unknown calibration domain {calibration!r}")


def quantize(model_name: str, calibration: str) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    from llmcompressor import oneshot  # noqa: PLC0415
    from llmcompressor.modifiers.quantization import QuantizationModifier  # noqa: PLC0415
    from llmcompressor.modifiers.transform.awq import AWQModifier  # noqa: PLC0415

    spec = get_model(model_name)
    save_dir = _checkpoint_dir(spec.name, calibration)

    print(f"Loading {spec.hf_repo_id} (fp16) for quantization...")
    model = AutoModelForCausalLM.from_pretrained(spec.hf_repo_id, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo_id)

    print(f"Loading {calibration} calibration dataset ({NUM_CALIBRATION_SAMPLES} samples)...")
    dataset = _load_calibration_dataset(calibration, tokenizer)

    recipe = [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
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

    manifest = build_manifest(
        {
            "model_name": spec.name,
            "hf_repo_id": spec.hf_repo_id,
            "quant": "gptq_awq_int4",
            "backend": "llmcompressor_awq",
            "calibration_domain": calibration,
            "calibration_dataset_id": _CODE_DATASET_ID if calibration == "code" else _CHAT_DATASET_ID,
            "num_calibration_samples": NUM_CALIBRATION_SAMPLES,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "recipe": ["AWQModifier(duo_scaling=both)", "QuantizationModifier(scheme=W4A16_ASYM, targets=Linear, ignore=lm_head)"],
        },
        repo_dir=_REPO_ROOT,
    )
    write_manifest(manifest, save_dir / "quantization_manifest.json")

    return save_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_name", help="ModelSpec.name, e.g. Qwen2.5-7B-Instruct (see models/registry.py)")
    parser.add_argument("--calibration", choices=["code", "chat"], required=True)
    args = parser.parse_args()

    save_dir = quantize(args.model_name, args.calibration)
    print(f"Done. Canonical loader.py path (not written automatically): {_QUANTIZED_DIR / (args.model_name + '-awq')}")
    print(f"This run's checkpoint: {save_dir}")


if __name__ == "__main__":
    main()
