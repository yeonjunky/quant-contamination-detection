"""`scripts/measure_sample_batch.py` with a fake model and device: no torch,
no GPU. The fakes stand in for CUDA's out-of-memory error and memory
counters; the script's own sampling path (`sample_item`) runs for real."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from qcd.data.schema import Dataset, Item
from qcd.models.registry import QWEN2_5_32B

_SCRIPT = Path(__file__).parents[1] / "scripts" / "measure_sample_batch.py"
_SPEC = importlib.util.spec_from_file_location("measure_sample_batch", _SCRIPT)
MEASURE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
# Registered first: the script's dataclass looks its module up by name.
sys.modules[_SPEC.name] = MEASURE
_SPEC.loader.exec_module(MEASURE)

GB = 10**9
_TEXT_SENTINEL = "GENERATED-TEXT-MUST-NOT-BE-STORED"


class FakeOutOfMemory(RuntimeError):
    pass


class FakeSample:
    def __init__(self, token_ids, token_logprobs, truncated_at_cap=False):
        self.text = _TEXT_SENTINEL
        self.token_ids = token_ids
        self.token_logprobs = token_logprobs
        self.truncated_at_cap = truncated_at_cap
        self.is_greedy = False


class FakeDevice:
    """Peak memory is 1 GB allocated and 1.2 GB reserved per sample row."""

    def __init__(self, total_bytes=80 * GB):
        self.total_bytes = total_bytes
        self.rows = 0
        self.releases = 0

    def reset_peak(self):
        self.rows = 0

    def peaks(self):
        return self.rows * GB, int(self.rows * 1.2 * GB)

    def release(self):
        self.releases += 1

    def is_out_of_memory(self, error):
        return isinstance(error, FakeOutOfMemory)

    def describe(self):
        return {"name": "fake", "total_memory_bytes": self.total_bytes}


class FakeModel:
    def __init__(self, device, *, max_rows, deterministic=True):
        self.device = device
        self.max_rows = max_rows
        self.deterministic = deterministic
        self.calls = 0
        self.revision = "fake-revision"

    def generate(self, item_id, prompt, *, temperature, sample_id):
        return FakeSample([1, 2, 3], [-0.1, -0.2, -0.3])

    def generate_samples(self, item_id, prompt, *, temperature, sample_ids):
        if len(sample_ids) > self.max_rows:
            raise FakeOutOfMemory("CUDA out of memory")
        self.device.rows = max(self.device.rows, len(sample_ids))
        self.calls += 1
        drift = 0.0 if self.deterministic else self.calls * 1e-6
        return [
            FakeSample([sample_id, 7], [-0.5 - drift, -0.25], truncated_at_cap=sample_id == 0)
            for sample_id in sample_ids
        ]


def _items():
    return [
        Item(item_id=f"lcb-{i}", dataset=Dataset.LCB_PRE, prompt="x" * (10 * i)) for i in range(1, 6)
    ]


def _run(tmp_path, model, device, *batch_sizes):
    record_path = MEASURE.main(
        ["--model", QWEN2_5_32B.name, "--quant", "bf16", "--n-items", "2",
         "--output-dir", str(tmp_path), "--batch-sizes", *map(str, batch_sizes)],
        load_model_fn=lambda spec, quant: model,
        load_items=_items,
        device=device,
        count_tokens_for=lambda model: len,
        save_pip_freeze=lambda output_dir: output_dir / "pip-freeze.txt",
    )
    return json.loads(record_path.read_text())


def _candidate(record, batch_size):
    return next(c for c in record["candidates"] if c["batch_size"] == batch_size)


def test_out_of_memory_on_a_large_size_still_tries_the_smaller_ones(tmp_path):
    device = FakeDevice()
    record = _run(tmp_path, FakeModel(device, max_rows=25), device, 10, 50, 25)

    assert [c["batch_size"] for c in record["candidates"]] == [50, 25, 10]
    largest = _candidate(record, 50)
    assert largest["out_of_memory"] is True
    assert largest["out_of_memory_at"] == "lcb-5"
    assert largest["peak_reserved_bytes"] is None
    assert largest["reproducible"] is None
    for size in (25, 10):
        fitted = _candidate(record, size)
        assert fitted["out_of_memory"] is False
        assert fitted["peak_allocated_bytes"] == size * GB
        assert fitted["reproducible"] is True
        assert len(fitted["seconds_per_item"]) == 2
    assert record["recommendation"]["batch_size"] == 25


def test_a_nondeterministic_model_fails_the_reproducibility_check(tmp_path):
    device = FakeDevice()
    record = _run(tmp_path, FakeModel(device, max_rows=50, deterministic=False), device, 50)

    candidate = _candidate(record, 50)
    assert candidate["reproducible"] is False
    assert "log-probabilities differ" in candidate["reproducibility_detail"]
    assert record["recommendation"]["batch_size"] is None


def test_peak_reserved_above_the_margin_is_not_recommended(tmp_path):
    # 50 rows reserve 60 GB and 25 rows 30 GB; on a 64 GB card the margin is
    # 57.6 GB, so 50 fits without OOM yet is not recommended.
    device = FakeDevice(total_bytes=64 * GB)
    record = _run(tmp_path, FakeModel(device, max_rows=50), device, 50, 25)

    assert _candidate(record, 50)["out_of_memory"] is False
    assert record["recommendation"]["batch_size"] == 25


def test_an_error_other_than_out_of_memory_is_not_swallowed(tmp_path):
    class BrokenModel(FakeModel):
        def generate_samples(self, *args, **kwargs):
            raise ValueError("shape mismatch")

    device = FakeDevice()
    with pytest.raises(ValueError, match="shape mismatch"):
        _run(tmp_path, BrokenModel(device, max_rows=50), device, 50)


def test_generated_length_summary_counts_greedy_and_samples(tmp_path):
    device = FakeDevice()
    record = _run(tmp_path, FakeModel(device, max_rows=50), device, 50)

    lengths = _candidate(record, 50)["generated_tokens"]
    # 2 items x (1 greedy of 3 tokens + 50 samples of 2 tokens); sample 0 of
    # each item is flagged as stopped at the cap.
    assert lengths["n_generations"] == 102
    assert lengths["max"] == 3 and lengths["median"] == 2
    assert lengths["n_at_512_cap"] == 2
    assert record["recommendation"]["worst_case_length_reached"] is True


def test_output_is_validation_only_and_holds_no_outcome_fields(tmp_path):
    device = FakeDevice()
    record = _run(tmp_path, FakeModel(device, max_rows=50), device, 50)
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert MEASURE._DEFAULT_OUTPUT == (
        Path(__file__).resolve().parents[2] / "data" / "raw" / "validation" / "sample_batch_measurement"
    )
    assert record["study_phase"] == manifest["study_phase"] == "engineering_validation"
    assert manifest["extra"]["gpu"] == {"name": "fake", "total_memory_bytes": 80 * GB}
    assert "git_dirty" in manifest and manifest["git_commit"]

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key.lower()
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    forbidden = ("pass", "score", "auc", "peakedness", "perplexity", "mink", "effect", "accuracy")
    offending = [k for k in [*keys(record), *keys(manifest["config"])] if any(f in k for f in forbidden)]
    assert offending == []
    for path in tmp_path.iterdir():
        if path.is_file():
            assert _TEXT_SENTINEL not in path.read_text()


def test_items_are_ranked_by_the_rendered_generation_prompt():
    # A functional item's generation prompt carries its starter code, so a
    # short statement with a long starter outranks a longer bare statement.
    short_with_starter = Item(
        item_id="functional", dataset=Dataset.LCB_PRE, prompt="short",
        metadata={"starter_code": "class Solution:\n" + "    pass\n" * 50},
    )
    long_stdin = Item(item_id="stdin", dataset=Dataset.LCB_POST, prompt="y" * 300)
    ranked = MEASURE.longest_prompts([long_stdin, short_with_starter], len, 1)
    assert ranked[0][0] == "functional"
