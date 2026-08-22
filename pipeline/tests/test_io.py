import json

import pandas as pd
import pytest

from qcd.data.schema import Dataset, Item
from qcd.io.manifest import build_manifest, config_hash, get_git_commit_hash, read_manifest, write_manifest
from qcd.io.raw_writer import RawDataWriter


# --- raw_writer --------------------------------------------------------------


def _items():
    return [
        Item(item_id="HumanEval/0", dataset=Dataset.HUMANEVAL, prompt="def f(): ...", metadata={"a": 1}),
        Item(item_id="q1", dataset=Dataset.LCB_PRE, prompt="solve this", difficulty="easy", metadata={"platform": "codeforces"}),
    ]


def test_write_items_roundtrip(tmp_path):
    writer = RawDataWriter(tmp_path)
    path = writer.write_items(_items())

    df = pd.read_parquet(path)
    assert len(df) == 2
    assert set(df["item_id"]) == {"HumanEval/0", "q1"}
    assert bool(df[df["item_id"] == "HumanEval/0"]["contamination_proxy"].iloc[0]) is True
    assert bool(df[df["item_id"] == "q1"]["contamination_proxy"].iloc[0]) is True
    # metadata round-trips as JSON
    meta = json.loads(df[df["item_id"] == "q1"]["metadata_json"].iloc[0])
    assert meta == {"platform": "codeforces"}


def test_add_generation_and_flush_roundtrip(tmp_path):
    writer = RawDataWriter(tmp_path)
    writer.add_generation(
        model="Qwen2.5-7B-Instruct", quant="bf16", item_id="x", sample_id=0, is_greedy=True,
        text="hello", token_ids=[1, 2, 3], token_logprobs=[-0.1, -0.2, -0.3],
        prompt_token_logprobs=[-1.1, -1.2],
        partial_pass_rate=1.0, passed=True, decoding_temperature=0.0,
        generation_seconds=2.0, prompt_scoring_seconds=0.5,
        sandbox_scoring_seconds=0.25,
    )
    assert writer.n_buffered_generations == 1

    written = writer.flush()
    df = pd.read_parquet(written["generations"])
    assert len(df) == 1
    assert list(df.iloc[0]["token_ids"]) == [1, 2, 3]
    assert list(df.iloc[0]["prompt_token_logprobs"]) == pytest.approx([-1.1, -1.2])
    assert df.iloc[0]["partial_pass_rate"] == pytest.approx(1.0)
    assert bool(df.iloc[0]["passed"]) is True
    assert df.iloc[0]["generation_seconds"] == pytest.approx(2.0)
    assert writer.n_buffered_generations == 0


def test_flush_can_write_atomic_bounded_parts(tmp_path):
    writer = RawDataWriter(tmp_path)
    writer.add_generation(
        model="m", quant="bf16", item_id="x", sample_id=0, is_greedy=True,
        text="x", token_ids=[1], token_logprobs=[-0.1],
    )
    first = writer.flush(part="m-bf16-00000")["generations"]
    assert first.name == "generations.m-bf16-00000.parquet"
    assert writer.n_buffered_generations == 0


def test_add_detector_score_and_flush_roundtrip(tmp_path):
    writer = RawDataWriter(tmp_path)
    writer.add_detector_score(
        model="Qwen2.5-7B-Instruct", quant="bnb_nf4", item_id="x", detector="cdd",
        score=0.42, threshold_used=0.01, source_sample_ids=[0, 1, 2],
    )
    assert writer.n_buffered_detector_scores == 1

    written = writer.flush()
    df = pd.read_parquet(written["detector_scores"])
    assert len(df) == 1
    assert df.iloc[0]["detector"] == "cdd"
    assert df.iloc[0]["score"] == pytest.approx(0.42)


def test_flush_with_nothing_buffered_writes_nothing(tmp_path):
    writer = RawDataWriter(tmp_path)
    assert writer.flush() == {}


def test_file_prefix_is_applied_to_all_parquet_names(tmp_path):
    writer = RawDataWriter(tmp_path, file_prefix="Qwen2.5-7B-Instruct")
    items_path = writer.write_items(_items())
    writer.add_generation(
        model="Qwen2.5-7B-Instruct", quant="bnb_nf4", item_id="x",
        sample_id=0, is_greedy=True, text="x", token_ids=[1],
        token_logprobs=[-0.1],
    )
    writer.add_detector_score(
        model="Qwen2.5-7B-Instruct", quant="bnb_nf4", item_id="x",
        detector="cdd", score=0.0,
    )
    written = writer.flush()

    assert items_path.name == "Qwen2.5-7B-Instruct_items.parquet"
    assert written["generations"].name == "Qwen2.5-7B-Instruct_generations.parquet"
    assert written["detector_scores"].name == "Qwen2.5-7B-Instruct_detector_scores.parquet"


# --- manifest ------------------------------------------------------------


def test_get_git_commit_hash_returns_something_in_this_repo():
    commit = get_git_commit_hash()
    assert commit is None or (isinstance(commit, str) and len(commit) == 40)


def test_config_hash_is_stable_under_key_order():
    h1 = config_hash({"a": 1, "b": 2})
    h2 = config_hash({"b": 2, "a": 1})
    assert h1 == h2


def test_config_hash_changes_with_content():
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_build_manifest_has_expected_fields():
    manifest = build_manifest({"model": "Qwen2.5-7B"}, seed=42)
    assert manifest.seed == 42
    assert manifest.config_hash == config_hash({"model": "Qwen2.5-7B"})
    assert manifest.config == {"model": "Qwen2.5-7B"}
    assert "numpy" in manifest.package_versions
    assert manifest.package_versions["numpy"] is not None  # numpy is a hard dependency, always installed
    # A GPU-only package not installed on this profile resolves to None, not a crash.
    assert "torch" in manifest.package_versions


def test_write_and_read_manifest_roundtrip(tmp_path):
    manifest = build_manifest({"x": 1}, seed=7)
    path = write_manifest(manifest, tmp_path / "manifest.json")

    loaded = read_manifest(path)
    assert loaded["seed"] == 7
    assert loaded["config_hash"] == manifest.config_hash
    assert loaded["config"] == {"x": 1}
