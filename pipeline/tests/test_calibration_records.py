"""Paper §4.3's AWQ calibration record and library-default record.

§4.3: "freeze one AWQ calibration artifact per model and record its dataset
revision, selected-row hashes, seed, tokenizer, quantizer recipe and software
versions ... Check candidates against all evaluation prompts and reference
solutions before use", and "Settings described as left at a library default
are not overridden by us; the resolved value is recorded in the run manifest."
"""

import importlib.util
import json
from pathlib import Path

import pytest

from qcd.io.manifest import (
    CALIBRATION_OVERLAP_NGRAM_SIZE, CALIBRATION_OVERLAP_REPORT_FILENAME,
    calibration_overlap_report, require_calibration_overlap_report,
    resolve_library_defaults, unresolved_library_defaults,
    write_calibration_overlap_report,
)

_SCRIPTS = Path(__file__).parents[1] / "scripts"

# 13+ tokens, so the n-gram rule applies rather than the short-text fallback.
_SHARED = (
    "def rolling_max(numbers):\n"
    "    result = []\n"
    "    current = float('-inf')\n"
    "    for value in numbers:\n"
    "        current = max(current, value)\n"
    "        result.append(current)\n"
    "    return result\n"
)
_UNRELATED = (
    "The weather forecast for the weekend mentions scattered showers in the "
    "afternoon, clearing overnight, with a light northerly breeze by Sunday "
    "morning across the whole valley.\n"
)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --- the overlap check ------------------------------------------------------


def test_overlap_check_finds_a_shared_ngram():
    report = calibration_overlap_report(
        calibration_texts=[_SHARED + "\n# extra calibration text\n"],
        evaluation_texts=[
            ("humaneval_prompt", "HumanEval/9", _SHARED),
            ("humaneval_prompt", "HumanEval/1", _UNRELATED),
        ],
    )
    assert report["n_evaluation_texts"] == 2
    assert report["n_overlapping_texts"] == 1
    overlap = report["overlaps"][0]
    assert overlap["key"] == "HumanEval/9"
    assert overlap["rule"] == "ngram"
    assert overlap["n_matching_ngrams"] >= 1


def test_overlap_check_reports_none_for_disjoint_text():
    report = calibration_overlap_report(
        calibration_texts=[_SHARED],
        evaluation_texts=[("mbppplus_prompt", "Mbpp/2", _UNRELATED)],
    )
    assert report["n_overlapping_texts"] == 0
    assert report["overlaps"] == []


def test_overlap_check_falls_back_to_containment_for_short_text():
    # Shorter than the n-gram window, so the n-gram rule cannot fire at all;
    # the containment rule is what keeps a short prompt from being skipped.
    short_prompt = "return the sum"
    report = calibration_overlap_report(
        calibration_texts=[f"please {short_prompt} of two numbers"],
        evaluation_texts=[("mbppplus_prompt", "Mbpp/3", short_prompt)],
    )
    assert report["n_overlapping_texts"] == 1
    assert report["overlaps"][0]["rule"] == "short-text-containment"


def test_overlap_report_documents_its_own_method():
    report = calibration_overlap_report(
        calibration_texts=[_SHARED], evaluation_texts=[("humaneval_prompt", "x", _SHARED)]
    )
    method = report["method"]
    assert method["ngram_size"] == CALIBRATION_OVERLAP_NGRAM_SIZE
    assert "NFKC" in method["normalization"]
    # A zero count is not a semantic-non-overlap claim (§4.3).
    assert "not detect paraphrase" in method["limitations"]
    assert report["sources"] == [
        {"source": "humaneval_prompt", "n_texts": 1, "n_overlapping": 1}
    ]


def test_overlap_report_counts_per_source():
    report = calibration_overlap_report(
        calibration_texts=[_SHARED],
        evaluation_texts=[
            ("livecodebench_question_content", "a", _SHARED),
            ("livecodebench_question_content", "b", _UNRELATED),
            ("humaneval_prompt", "c", _UNRELATED),
        ],
    )
    assert report["sources"] == [
        {"source": "humaneval_prompt", "n_texts": 1, "n_overlapping": 0},
        {"source": "livecodebench_question_content", "n_texts": 2, "n_overlapping": 1},
    ]


# --- the loader's refusal ---------------------------------------------------


def test_require_overlap_report_refuses_a_checkpoint_without_one(tmp_path):
    with pytest.raises(FileNotFoundError, match=CALIBRATION_OVERLAP_REPORT_FILENAME):
        require_calibration_overlap_report(tmp_path)


def test_require_overlap_report_refuses_a_malformed_report(tmp_path):
    (tmp_path / CALIBRATION_OVERLAP_REPORT_FILENAME).write_text(
        json.dumps({"method": {}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing fields"):
        require_calibration_overlap_report(tmp_path)


def test_require_overlap_report_refuses_a_check_that_covered_nothing(tmp_path):
    write_calibration_overlap_report(
        calibration_overlap_report(calibration_texts=[_SHARED], evaluation_texts=[]),
        tmp_path,
    )
    with pytest.raises(ValueError, match="0 evaluation texts"):
        require_calibration_overlap_report(tmp_path)


def test_require_overlap_report_accepts_a_written_report(tmp_path):
    written = write_calibration_overlap_report(
        calibration_overlap_report(
            calibration_texts=[_SHARED],
            evaluation_texts=[("humaneval_prompt", "x", _UNRELATED)],
        ),
        tmp_path,
    )
    assert written.name == CALIBRATION_OVERLAP_REPORT_FILENAME
    assert require_calibration_overlap_report(tmp_path)["n_evaluation_texts"] == 1


def test_awq_load_refused_when_the_overlap_check_is_missing(tmp_path, monkeypatch):
    import qcd.models.loader as loader
    from qcd.config import Quant
    from qcd.models.registry import QWEN2_5_7B

    monkeypatch.setattr(loader, "_QUANTIZED_DIR", tmp_path)
    # A checkpoint directory that exists but was never overlap-checked: the
    # refusal must come from the missing report, not from the missing files,
    # and before any heavy import.
    (tmp_path / f"{QWEN2_5_7B.name}-awq").mkdir()
    with pytest.raises(FileNotFoundError, match=CALIBRATION_OVERLAP_REPORT_FILENAME):
        loader.load_model(QWEN2_5_7B, Quant.GPTQ_AWQ_INT4, mock=False)


# --- §4.3's resolved library defaults --------------------------------------


def test_every_library_default_is_either_resolved_or_explained():
    resolved = resolve_library_defaults()
    assert set(resolved) == {
        "bnb_llm_int8_outlier_threshold",
        "bnb_4bit_block_size",
        "bnb_modules_not_converted",
        "awq_group_size",
        "mbppplus_dataset_version",
    }
    for name, entry in resolved.items():
        assert set(entry) == {"value", "source", "unavailable_reason"}
        # Never both, never neither: a missing value always carries its reason,
        # and nothing is filled in from memory.
        has_value = entry["value"] is not None
        has_reason = entry["unavailable_reason"] is not None
        assert has_value != has_reason, name
        if has_value:
            assert entry["source"]


def test_unresolved_entries_are_reportable():
    resolved = {
        "a": {"value": 1, "source": "s", "unavailable_reason": None},
        "b": {"value": None, "source": None, "unavailable_reason": "not installed"},
    }
    assert unresolved_library_defaults(resolved) == {"b": "not installed"}


def test_mbppplus_dataset_version_resolves_from_evalplus():
    entry = resolve_library_defaults()["mbppplus_dataset_version"]
    assert entry["value"]
    assert entry["source"] == "evalplus.data.mbpp.MBPP_PLUS_VERSION"


def test_int8_outlier_threshold_resolves_from_transformers():
    entry = resolve_library_defaults()["bnb_llm_int8_outlier_threshold"]
    assert isinstance(entry["value"], float)
    assert "BitsAndBytesConfig" in entry["source"]


def test_skip_list_resolution_reads_the_loaded_model():
    class _Quantizer:
        modules_to_not_convert = ["lm_head"]

    class _Model:
        hf_quantizer = _Quantizer()

    entry = resolve_library_defaults(model=_Model())["bnb_modules_not_converted"]
    assert entry["value"] == ["lm_head"]
    assert entry["unavailable_reason"] is None


# --- the quantization script's own records ---------------------------------


def test_calibration_row_hashes_are_deterministic_and_ordered():
    quantize_model = _load_script("quantize_model")
    first = quantize_model.calibration_row_hashes(["a", "b"])
    assert first == quantize_model.calibration_row_hashes(["a", "b"])
    assert first["selected_rows_sha256"] != quantize_model.calibration_row_hashes(
        ["b", "a"]
    )["selected_rows_sha256"]
    assert len(first["row_sha256"]) == 2


def test_quantize_script_is_fixed_to_one_code_calibration():
    quantize_model = _load_script("quantize_model")
    assert quantize_model.CALIBRATION_DATASET_ID == "flytech/python-codes-25k"
    assert quantize_model.CALIBRATION_SHUFFLE_SEED == 42
    # E-F4: no calibration-variant comparison and no "copy the winner" step.
    source = (_SCRIPTS / "quantize_model.py").read_text(encoding="utf-8")
    assert "--calibration" not in source
    assert "winner" not in source.lower()


def test_quantize_script_writes_the_loader_canonical_path():
    quantize_model = _load_script("quantize_model")
    from qcd.models.loader import _quantized_checkpoint_dir
    from qcd.models.registry import QWEN2_5_7B

    assert quantize_model._checkpoint_dir(QWEN2_5_7B.name).name == (
        _quantized_checkpoint_dir(QWEN2_5_7B).name
    )
