"""`scripts/measure_sample_batch.py` with a fake model and device: no torch,
no GPU. The fakes stand in for CUDA's out-of-memory error and memory
counters; the script's own sampling path (`sample_item`) runs for real.
The stop-token suppression itself runs on a real tiny model in
tests/test_real_model_sampling.py."""

import importlib.util
import json
import math
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
_CAP = 512
_EOS = 2
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
    """Reserved memory is 1.2 x the largest allocation since the last reset."""

    def __init__(self, total_bytes=80 * GB):
        self.total_bytes = total_bytes
        self.peak = 0

    def touch(self, n_bytes):
        self.peak = max(self.peak, n_bytes)

    def reset_peak(self):
        self.peak = 0

    def peaks(self):
        return self.peak, int(self.peak * 1.2)

    def release(self):
        pass

    def is_out_of_memory(self, error):
        return isinstance(error, FakeOutOfMemory)

    def describe(self):
        return {"name": "fake", "total_memory_bytes": self.total_bytes}


class FakeModel:
    """1 GB per sample row with normal decoding, 1.5 GB per row while stop
    tokens are suppressed, and `scoring_bytes` for the prompt-scoring pass."""

    def __init__(self, device, *, max_rows, deterministic=True, logprob_offset=0.0,
                 scoring_bytes=0, honours_processors=True):
        self.device = device
        self.max_rows = max_rows
        self.deterministic = deterministic
        self.logprob_offset = logprob_offset
        self.scoring_bytes = scoring_bytes
        self.honours_processors = honours_processors
        self.calls = 0
        self.revision = "fake-revision"
        self.eos_token_ids = frozenset({_EOS})
        self.extra_logits_processors = ()
        self.processors_seen = []
        self.scored = []

    def _forced(self):
        self.processors_seen.append(self.extra_logits_processors)
        return bool(self.extra_logits_processors) and self.honours_processors

    def generate(self, item_id, prompt, *, temperature, sample_id):
        if self._forced():
            return FakeSample([5] * _CAP, [-0.1] * _CAP, truncated_at_cap=True)
        return FakeSample([1, 2, 3], [-0.1, -0.2, -0.3])

    def generate_samples(self, item_id, prompt, *, temperature, sample_ids):
        if len(sample_ids) > self.max_rows:
            raise FakeOutOfMemory("CUDA out of memory")
        forced = self._forced()
        self.device.touch(int(len(sample_ids) * GB * (1.5 if forced else 1.0)))
        if forced:
            return [FakeSample([5] * _CAP, [-0.1] * _CAP, truncated_at_cap=True) for _ in sample_ids]
        self.calls += 1
        drift = 0.0 if self.deterministic else self.calls * 1e-6
        return [
            FakeSample([sample_id, 7], [-0.5 - drift - self.logprob_offset, -0.25],
                       truncated_at_cap=sample_id == 0)
            for sample_id in sample_ids
        ]

    def score_prompt_detail(self, item_id, prompt):
        self.scored.append((item_id, prompt, bool(self.extra_logits_processors)))
        self.device.touch(self.scoring_bytes)
        return object()


def _items():
    return [
        Item(item_id=f"lcb-{i}", dataset=Dataset.LCB_PRE, prompt="x" * (10 * i)) for i in range(1, 6)
    ]


def _run_path(tmp_path, model, device, *batch_sizes, n_items=2, compare_to=None):
    argv = ["--model", QWEN2_5_32B.name, "--quant", "bf16", "--n-items", str(n_items),
            "--output-dir", str(tmp_path), "--batch-sizes", *map(str, batch_sizes)]
    if compare_to is not None:
        argv += ["--compare-to", str(compare_to)]
    return MEASURE.main(
        argv,
        load_model_fn=lambda spec, quant: model,
        load_items=_items,
        device=device,
        count_tokens_for=lambda model: len,
        save_pip_freeze=lambda output_dir: output_dir / "pip-freeze.txt",
    )


def _run(tmp_path, model, device, *batch_sizes, **kwargs):
    return json.loads(_run_path(tmp_path, model, device, *batch_sizes, **kwargs).read_text())


def _candidate(record, batch_size):
    return next(c for c in record["candidates"] if c["batch_size"] == batch_size)


def test_out_of_memory_on_a_large_size_still_tries_the_smaller_ones(tmp_path):
    device = FakeDevice()
    record = _run(tmp_path, FakeModel(device, max_rows=25), device, 10, 50, 25)

    assert [c["batch_size"] for c in record["candidates"]] == [50, 25, 10]
    largest = _candidate(record, 50)
    assert largest["out_of_memory"] is True
    assert largest["out_of_memory_at"] == "normal_decoding:lcb-5"
    assert largest["normal_decoding"] is None and largest["forced_length"] is None
    assert largest["reproducible"] is None
    for size in (25, 10):
        fitted = _candidate(record, size)
        assert fitted["out_of_memory"] is False
        assert fitted["normal_decoding"]["decoding"] == "normal_decoding"
        assert fitted["normal_decoding"]["peak_allocated_bytes"] == size * GB
        assert fitted["forced_length"]["decoding"] == "forced_length"
        assert fitted["forced_length"]["peak_allocated_bytes"] == int(size * 1.5 * GB)
        assert fitted["forced_length_reached"] is True
        assert fitted["reproducible"] is True
        assert len(fitted["normal_decoding"]["seconds_per_item"]) == 2
        assert len(fitted["forced_length"]["seconds_per_item"]) == 2
    assert record["recommendation"]["batch_size"] == 25
    assert record["recommendation"]["memory_from"] == "forced_length"


def test_a_nondeterministic_model_fails_the_reproducibility_check(tmp_path):
    device = FakeDevice()
    record = _run(tmp_path, FakeModel(device, max_rows=50, deterministic=False), device, 50)

    candidate = _candidate(record, 50)
    assert candidate["reproducible"] is False
    assert "log-probabilities differ" in candidate["reproducibility_detail"]
    assert record["recommendation"]["batch_size"] is None
    assert MEASURE.exit_code(record) == 1


def test_the_margin_is_judged_on_the_forced_length_peak(tmp_path):
    # 64 GB card, margin 57.6 GB reserved. 40 rows reserve 48 GB with normal
    # decoding but 72 GB forced to 512 tokens; 25 rows reserve 45 GB forced.
    device = FakeDevice(total_bytes=64 * GB)
    record = _run(tmp_path, FakeModel(device, max_rows=50), device, 40, 25)

    forty = _candidate(record, 40)
    assert forty["out_of_memory"] is False
    assert forty["normal_decoding"]["peak_reserved_bytes"] == 48 * GB
    assert forty["forced_length"]["peak_reserved_bytes"] == 72 * GB
    assert record["recommendation"]["batch_size"] == 25


def test_the_prompt_scoring_pass_is_in_the_measured_peak(tmp_path):
    # Generation alone peaks at 15 GB forced; the scoring pass needs 70 GB,
    # 84 GB reserved, above the 72 GB margin of an 80 GB card.
    device = FakeDevice()
    model = FakeModel(device, max_rows=50, scoring_bytes=70 * GB)
    record = _run(tmp_path, model, device, 10)

    candidate = _candidate(record, 10)
    assert candidate["normal_decoding"]["peak_allocated_bytes"] == 70 * GB
    assert candidate["forced_length"]["peak_allocated_bytes"] == 70 * GB
    assert record["recommendation"]["batch_size"] is None
    # Each measured item is scored on its own prompt, once per pass, as the
    # main run does per item.
    longest = [item for item in _items() if item.item_id in ("lcb-5", "lcb-4")]
    assert sorted(model.scored) == sorted(
        [(item.item_id, item.prompt, forced) for item in longest for forced in (False, True)]
    )


def test_no_recommendation_tells_the_operator_to_try_smaller_sizes(tmp_path, capsys):
    device = FakeDevice()
    record = _run(tmp_path, FakeModel(device, max_rows=1), device, 5, 2)

    assert record["recommendation"]["batch_size"] is None
    assert "--batch-sizes 4 3 2 1" in record["recommendation"]["reason"]
    assert "--batch-sizes 4 3 2 1" in capsys.readouterr().out
    assert MEASURE.exit_code(record) == 1


def test_a_forced_pass_that_did_not_reach_the_cap_is_not_recommended(tmp_path):
    device = FakeDevice()
    record = _run(tmp_path, FakeModel(device, max_rows=50, honours_processors=False), device, 10)

    assert _candidate(record, 10)["forced_length_reached"] is False
    assert record["recommendation"]["batch_size"] is None


def test_stop_tokens_are_suppressed_only_during_the_forced_pass(tmp_path):
    device = FakeDevice()
    model = FakeModel(device, max_rows=50)
    _run(tmp_path, model, device, 25)

    assert model.extra_logits_processors == ()
    # Greedy + 2 sample chunks per item; 2 items normal, 1 repeat, 2 items forced.
    normal, forced = model.processors_seen[:9], model.processors_seen[9:]
    assert normal == [()] * 9 and len(forced) == 6
    for processors in forced:
        [suppressor] = processors
        assert isinstance(suppressor, MEASURE.StopTokenSuppressor)
        assert suppressor.token_ids == [_EOS]


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

    candidate = _candidate(record, 50)
    lengths = candidate["normal_decoding"]["generated_tokens"]
    # 2 items x (1 greedy of 3 tokens + 50 samples of 2 tokens); sample 0 of
    # each item is flagged as stopped at the cap.
    assert lengths["n_generations"] == 102
    assert lengths["max"] == 3 and lengths["median"] == 2
    assert lengths["n_at_512_cap"] == 2
    forced = candidate["forced_length"]["generated_tokens"]
    assert forced["min"] == forced["max"] == _CAP and forced["cap_rate"] == 1.0


def test_a_second_process_with_identical_generations_matches(tmp_path):
    device = FakeDevice()
    first = _run_path(tmp_path / "first", FakeModel(device, max_rows=50), device, 50, 25)
    second = _run(tmp_path / "second", FakeModel(device, max_rows=50), device, 50, 25, compare_to=first)

    earlier = json.loads(first.read_text())
    for size in (50, 25):
        digest = _candidate(second, size)["generations_sha256"]
        assert len(digest) == 64 and digest == _candidate(earlier, size)["generations_sha256"]
    assert second["comparison"]["all_match"] is True
    assert second["comparison"]["per_batch_size"] == {"50": True, "25": True}
    assert MEASURE.exit_code(second) == 0


def test_a_second_process_with_different_generations_fails(tmp_path):
    device = FakeDevice()
    first = _run_path(tmp_path / "first", FakeModel(device, max_rows=50), device, 50, 25)
    # Each process is reproducible within itself, but the log-probabilities
    # differ between them.
    second = _run(tmp_path / "second", FakeModel(device, max_rows=50, logprob_offset=1e-9),
                  device, 50, 25, compare_to=first)

    assert _candidate(second, 25)["reproducible"] is True
    assert second["recommendation"]["batch_size"] == 25
    assert second["comparison"]["all_match"] is False
    assert second["comparison"]["per_batch_size"] == {"50": False, "25": False}
    assert MEASURE.exit_code(second) == 1


@pytest.mark.parametrize("change", [
    {"n_items": 3},
    {"batch_sizes": (10,)},
])
def test_a_comparison_with_nothing_comparable_fails(tmp_path, change):
    device = FakeDevice()
    first = _run_path(tmp_path / "first", FakeModel(device, max_rows=50), device, 50, 25)
    second = _run(tmp_path / "second", FakeModel(device, max_rows=50), device,
                  *change.get("batch_sizes", (50, 25)), n_items=change.get("n_items", 2),
                  compare_to=first)

    assert second["comparison"]["all_match"] is False
    assert second["comparison"]["reason"]
    assert MEASURE.exit_code(second) == 1


def test_the_digest_sees_one_ulp_in_one_log_probability():
    item = MEASURE.MeasuredItem("lcb-1", "p", "p", 1)

    class Generations:
        def __init__(self, logprob):
            self.greedy = FakeSample([1, 2], [-0.5, -0.25])
            self.samples = [FakeSample([3], [logprob])]

    base = MEASURE.generations_sha256([item], [Generations(-0.75)])
    assert base == MEASURE.generations_sha256([item], [Generations(-0.75)])
    assert base != MEASURE.generations_sha256([item], [Generations(math.nextafter(-0.75, 0.0))])


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
    [ranked] = MEASURE.longest_prompts([long_stdin, short_with_starter], len, 1)
    assert ranked.item_id == "functional"
    assert ranked.scoring_prompt == "short"


def test_only_the_measurement_script_sets_the_extra_logits_processors_hook():
    pipeline = Path(__file__).parents[1]
    users = sorted(
        str(path.relative_to(pipeline))
        for path in [*(pipeline / "src").rglob("*.py"), *(pipeline / "scripts").rglob("*.py")]
        if "extra_logits_processors" in path.read_text(encoding="utf-8")
    )
    assert users == ["scripts/measure_sample_batch.py", "src/qcd/models/loader.py"]
