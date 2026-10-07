import dataclasses
import json
import subprocess

import pandas as pd
import pytest

from qcd.data.schema import (
    CorpusEvidenceFamily, CorpusReferenceRecord, CorpusReferenceStatus, Dataset, Item,
    corpus_reference_from_json,
)
from qcd.io.manifest import (
    StudyPhase, build_manifest, config_hash, get_git_commit_hash, read_manifest,
    require_clean_checkout, write_manifest,
)
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
    # unmeasured corpus axis is an empty list, not a null
    assert json.loads(df["corpus_reference_json"].iloc[0]) == []


def test_items_parquet_stores_the_per_model_corpus_axis(tmp_path):
    """Paper §4.2 stores corpus evidence per model on a separate,
    non-exclusive axis; §5 step 5 reports each method family separately. The
    single `tracer_label` float this column replaced could hold neither."""
    item = dataclasses.replace(
        _items()[0],
        corpus_reference=(
            CorpusReferenceRecord(
                model="Olmo3-7B-Instruct",
                method_family=CorpusEvidenceFamily.INSTANCE_STRING.value,
                status=CorpusReferenceStatus.CONFIRMED_MATCH,
                corpus="allenai/dolma3_mix-6T-1025-7B",
                corpus_revision="2ca900fbe14e86c5c83d064d9f0882f1c0b8c05b",
                stage="pretraining",
            ),
            CorpusReferenceRecord(
                model="Qwen2.5-7B-Instruct",
                method_family=CorpusEvidenceFamily.INSTANCE_STRING.value,
                status=CorpusReferenceStatus.NOT_OBSERVABLE,
            ),
        ),
    )
    path = RawDataWriter(tmp_path).write_items([item])
    stored = pd.read_parquet(path)["corpus_reference_json"].iloc[0]

    assert corpus_reference_from_json(stored) == item.corpus_reference
    assert "tracer_label" not in pd.read_parquet(path).columns


def _encode_private_tests(cases: list[dict]) -> str:
    import base64
    import pickle
    import zlib
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()


def test_items_parquet_stores_test_payloads_as_hash_and_count(tmp_path):
    import hashlib

    public = json.dumps([{"input": "PUBLIC-IN", "output": "1", "testtype": "stdin"}])
    private = _encode_private_tests([
        {"input": f"PRIVATE-IN-{i}", "output": "2", "testtype": "stdin"} for i in range(3)
    ])
    lcb = Item(
        item_id="q1", dataset=Dataset.LCB_PRE, prompt="solve",
        metadata={
            "platform": "atcoder", "func_name": None,
            "public_test_cases": public, "private_test_cases": private,
        },
    )
    evalplus = Item(
        item_id="HumanEval/0", dataset=Dataset.HUMANEVAL, prompt="def f(x):",
        metadata={
            "evalplus_dataset_name": "humaneval",
            "evalplus_problem": {
                "entry_point": "f", "canonical_solution": "    return x",
                "base_input": [["BASE-IN"]], "plus_input": [["PLUS-IN"], ["PLUS-IN-2"]],
            },
        },
    )
    path = RawDataWriter(tmp_path).write_items([lcb, evalplus])

    stored = pd.read_parquet(path).set_index("item_id")["metadata_json"]
    assert "-IN" not in "".join(stored)
    lcb_meta = json.loads(stored["q1"])
    assert lcb_meta["platform"] == "atcoder"
    assert lcb_meta["public_test_cases"] == {
        "sha256": hashlib.sha256(public.encode()).hexdigest(), "n_tests": 1,
    }
    assert lcb_meta["private_test_cases"] == {
        "sha256": hashlib.sha256(private.encode()).hexdigest(), "n_tests": 3,
    }
    problem = json.loads(stored["HumanEval/0"])["evalplus_problem"]
    assert problem["entry_point"] == "f"
    assert problem["base_input"]["n_tests"] == 1
    assert problem["plus_input"]["n_tests"] == 2
    # The run itself still scores against the full in-memory item.
    assert lcb.metadata["private_test_cases"] == private


def test_items_parquet_is_not_rewritten_when_unchanged(tmp_path):
    writer = RawDataWriter(tmp_path)
    path = writer.write_items(_items())
    first = path.stat()
    writer.write_items(_items())
    assert path.stat().st_ino == first.st_ino
    assert path.stat().st_mtime_ns == first.st_mtime_ns

    changed = _items()[:1]
    writer.write_items(changed)
    assert list(pd.read_parquet(path)["item_id"]) == ["HumanEval/0"]


def _lcb_release_v6_is_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        from qcd.data import livecodebench
    except ImportError:
        return False
    return all(
        isinstance(
            try_to_load_from_cache(
                livecodebench._REPO_ID, filename,
                revision=livecodebench.REPO_REVISION, repo_type="dataset",
            ),
            str,
        )
        for filename in livecodebench._RELEASE_FILES["release_v6"]
    )


@pytest.mark.skipif(not _lcb_release_v6_is_cached(), reason="LiveCodeBench release_v6 not in the HF cache")
def test_real_lcb_release_v6_items_write_a_small_items_parquet(tmp_path):
    """Storing the raw test payloads made this column about 4.5 GB, and
    pyarrow refused it, so every main-study cell died before loading a model."""
    import datetime as dt

    from qcd.data.livecodebench import load_livecodebench_split

    pre, post = load_livecodebench_split(dt.datetime(2025, 1, 1))
    items = pre + post
    assert len(items) == 1055

    path = RawDataWriter(tmp_path).write_items(items)

    assert path.stat().st_size < 50_000_000
    stored = pd.read_parquet(path)
    assert len(stored) == 1055
    for metadata_json in stored["metadata_json"]:
        metadata = json.loads(metadata_json)
        for key in ("public_test_cases", "private_test_cases"):
            assert metadata[key] is None or set(metadata[key]) == {"sha256", "n_tests"}


def test_write_model_item_labels_roundtrip(tmp_path):
    writer = RawDataWriter(tmp_path)
    path = writer.write_model_item_labels([{
        "model": "model",
        "item_id": "item",
        "dataset": "lcb_pre",
        "publication_date": "2023-06-01",
        "primary_first_post_date": "2024-01-01",
        "sensitivity_first_post_date": "2023-04-01",
        "shared_control_start_date": "2025-01-01",
        "primary_label": "possible-exposure",
        "sensitivity_label": "clean-by-model-cutoff",
        "boundary_ambiguous": True,
    }])
    row = pd.read_parquet(path).iloc[0]
    assert row["primary_label"] == "possible-exposure"
    assert row["publication_date"] == "2023-06-01"
    assert bool(row["boundary_ambiguous"]) is True


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


def test_add_generation_roundtrips_decoding_and_prompt_provenance(tmp_path):
    # Paper §4.4's record list for the probability detectors — "target text,
    # token boundaries, truncation, chat template, tokenizer/checkpoint
    # revisions and decoding settings".
    writer = RawDataWriter(tmp_path)
    writer.add_generation(
        model="Qwen2.5-7B-Instruct", quant="bnb_nf4", item_id="x", sample_id=0,
        is_greedy=True, text="hello", token_ids=[1, 2], token_logprobs=[-0.1, -0.2],
        prompt_token_logprobs=[-1.1, -1.2],
        truncated_at_cap=True, max_new_tokens=512,
        decoding_settings_id="0123456789abcdef",
        chat_template_id="fedcba9876543210",
        prompt_chat_template_applied=True,
        prompt_target_text_sha256="a" * 64,
        prompt_target_char_span=(31, 47),
        prompt_target_token_indices=[9, 10, 11],
        prompt_rendered_char_length=64,
        model_revision="model-rev", tokenizer_revision="tokenizer-rev",
    )
    row = pd.read_parquet(writer.flush()["generations"]).iloc[0]

    assert bool(row["truncated_at_cap"]) is True
    assert int(row["max_new_tokens"]) == 512
    assert row["decoding_settings_id"] == "0123456789abcdef"
    assert row["chat_template_id"] == "fedcba9876543210"
    assert bool(row["prompt_chat_template_applied"]) is True
    assert row["prompt_target_text_sha256"] == "a" * 64
    assert list(row["prompt_target_char_span"]) == [31, 47]
    assert list(row["prompt_target_token_indices"]) == [9, 10, 11]
    assert int(row["prompt_rendered_char_length"]) == 64
    assert row["model_revision"] == "model-rev"
    assert row["tokenizer_revision"] == "tokenizer-rev"


def test_generation_provenance_fields_are_optional(tmp_path):
    # A sample row (no fixed-prompt scoring pass) omits every prompt_* field,
    # and a caller that predates them keeps working — they default to None.
    writer = RawDataWriter(tmp_path)
    writer.add_generation(
        model="m", quant="bf16", item_id="x", sample_id=1, is_greedy=False,
        text="x", token_ids=[1], token_logprobs=[-0.1],
    )
    row = pd.read_parquet(writer.flush()["generations"]).iloc[0]

    assert row["truncated_at_cap"] is None
    assert row["prompt_target_char_span"] is None
    assert row["chat_template_id"] is None


def test_write_chat_templates_roundtrip(tmp_path):
    writer = RawDataWriter(tmp_path)
    path = writer.write_chat_templates([{
        "model": "Qwen2.5-7B-Instruct",
        "quant": "bf16",
        "chat_template_id": "fedcba9876543210",
        "chat_template_applied": True,
        "chat_template": "{% for message in messages %}...{% endfor %}",
        "tokenizer_revision": "tokenizer-rev",
        "model_revision": "model-rev",
    }], part="Qwen2.5-7B-Instruct-bf16")

    assert path.name == "chat_templates.Qwen2.5-7B-Instruct-bf16.parquet"
    row = pd.read_parquet(path).iloc[0]
    assert row["chat_template_id"] == "fedcba9876543210"
    assert "{% for message in messages %}" in row["chat_template"]


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
        cdd_threshold_length=37, cdd_n_empty_samples=1, cdd_greedy_empty=False,
    )
    writer.add_detector_score(
        model="Qwen2.5-7B-Instruct", quant="bnb_nf4", item_id="x",
        detector="perplexity", score=-1.5,
    )
    assert writer.n_buffered_detector_scores == 2

    written = writer.flush()
    df = pd.read_parquet(written["detector_scores"]).set_index("detector")
    assert len(df) == 2
    assert df.loc["cdd", "score"] == pytest.approx(0.42)
    assert df.loc["cdd", "cdd_threshold_length"] == 37
    assert df.loc["cdd", "cdd_n_empty_samples"] == 1
    assert df.loc["cdd", "cdd_greedy_empty"] == False  # noqa: E712
    assert df.loc["perplexity", ["cdd_threshold_length", "cdd_n_empty_samples",
                                 "cdd_greedy_empty"]].isna().all()


def test_a_part_is_complete_only_once_both_of_its_files_exist(tmp_path):
    writer = RawDataWriter(tmp_path)
    writer.add_generation(
        model="m", quant="bf16", item_id="x", sample_id=0, is_greedy=True,
        text="x", token_ids=[1], token_logprobs=[-0.1],
    )
    writer.add_detector_score(model="m", quant="bf16", item_id="x", detector="cdd", score=1.0)
    assert not writer.has_part("m-bf16-00000")
    written = writer.flush(part="m-bf16-00000")
    assert writer.has_part("m-bf16-00000")
    # An interruption between the two atomic writes leaves only generations.
    written["detector_scores"].unlink()
    assert not writer.has_part("m-bf16-00000")


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
    manifest = build_manifest(
        {"model": "Qwen2.5-7B"}, study_phase=StudyPhase.MAIN_STUDY, seed=42
    )
    assert manifest.study_phase == "main_study"
    assert manifest.seed == 42
    assert manifest.config_hash == config_hash({"model": "Qwen2.5-7B"})
    assert manifest.config == {"model": "Qwen2.5-7B"}
    assert "numpy" in manifest.package_versions
    assert manifest.package_versions["numpy"] is not None  # numpy is a hard dependency, always installed
    # A GPU-only package not installed on this profile resolves to None, not a crash.
    assert "torch" in manifest.package_versions


def test_write_and_read_manifest_roundtrip(tmp_path):
    manifest = build_manifest({"x": 1}, study_phase=StudyPhase.MAIN_STUDY, seed=7)
    path = write_manifest(manifest, tmp_path / "manifest.json")

    loaded = read_manifest(path)
    assert loaded["study_phase"] == "main_study"
    assert loaded["seed"] == 7
    assert loaded["config_hash"] == manifest.config_hash
    assert loaded["config"] == {"x": 1}


def test_a_legacy_items_parquet_still_loads_through_the_analysis_reader(tmp_path):
    """An `items.parquet` written before the corpus axis existed carries a
    `tracer_label` float and no `corpus_reference_json`. The analysis reader
    must still open it: this change alters what is written, not what can be
    read."""
    from qcd.analysis import study_inputs

    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame([{
        "item_id": "q1", "dataset": "lcb_pre", "prompt": "solve this",
        "difficulty": "easy", "contamination_proxy": True,
        "tracer_label": None, "release_version": "release_v6", "metadata_json": "{}",
    }]).to_parquet(raw / "items.parquet", index=False)
    pd.DataFrame([{
        "model": "m", "item_id": "q1", "dataset": "lcb_pre",
        "primary_label": "possible-exposure",
    }]).to_parquet(raw / "model_item_labels.parquet", index=False)
    pd.DataFrame([{
        "model": "m", "quant": "bf16", "item_id": "q1", "detector": "cdd", "score": 0.5,
    }]).to_parquet(raw / "detector_scores.parquet", index=False)
    pd.DataFrame([{
        "model": "m", "quant": "bf16", "item_id": "q1", "sample_id": 0, "is_greedy": True,
    }]).to_parquet(raw / "generations.parquet", index=False)

    tables = study_inputs.load_raw_tables(raw)
    assert list(tables.items["item_id"]) == ["q1"]
    assert "corpus_reference_json" not in tables.items.columns
    assert corpus_reference_from_json(tables.items["tracer_label"].iloc[0]) == ()


def _git_repo_with_one_commit(path):
    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=path, check=True, capture_output=True,
        )
    git("init", "-q")
    (path / "code.py").write_text("x = 1\n")
    git("add", "code.py")
    git("commit", "-q", "-m", "init")
    return path


def test_manifest_records_a_clean_checkout(tmp_path):
    repo = _git_repo_with_one_commit(tmp_path)
    manifest = build_manifest({"x": 1}, study_phase=StudyPhase.MAIN_STUDY, repo_dir=repo)
    assert len(manifest.git_commit) == 40
    assert manifest.git_dirty is False
    assert manifest.git_tracked_diff_sha256 is None
    assert require_clean_checkout(repo) == manifest.git_commit


def test_manifest_records_uncommitted_tracked_changes_outside_the_config_hash(tmp_path):
    repo = _git_repo_with_one_commit(tmp_path)
    clean = build_manifest({"x": 1}, study_phase=StudyPhase.MAIN_STUDY, repo_dir=repo)
    (repo / "code.py").write_text("x = 2\n")
    first = build_manifest({"x": 1}, study_phase=StudyPhase.MAIN_STUDY, repo_dir=repo)
    (repo / "code.py").write_text("x = 3\n")
    second = build_manifest({"x": 1}, study_phase=StudyPhase.MAIN_STUDY, repo_dir=repo)

    assert first.git_dirty is True
    assert len(first.git_tracked_diff_sha256) == 64
    assert first.git_tracked_diff_sha256 != second.git_tracked_diff_sha256
    # A dirty tree is recorded, but must not read as a different run configuration.
    assert first.config_hash == clean.config_hash
    with pytest.raises(RuntimeError, match="uncommitted"):
        require_clean_checkout(repo)


def test_untracked_files_do_not_make_the_checkout_dirty(tmp_path):
    repo = _git_repo_with_one_commit(tmp_path)
    (repo / "notes.md").write_text("scratch\n")
    assert build_manifest({}, study_phase=StudyPhase.MAIN_STUDY, repo_dir=repo).git_dirty is False


def test_require_clean_checkout_refuses_code_outside_git(tmp_path):
    with pytest.raises(RuntimeError, match="git"):
        require_clean_checkout(tmp_path)
