"""E-F14: the scan entry point has to say which checkpoint a corpus belongs to.

Both cases below stop inside argument handling, before any Hub access.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "search_olmo_corpus.py"
SPEC = importlib.util.spec_from_file_location("search_olmo_corpus", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _main(monkeypatch, tmp_path, *args):
    monkeypatch.setattr(
        "sys.argv",
        ["search_olmo_corpus.py", "--output", str(tmp_path / "out.jsonl"), *args],
    )
    MODULE.main()


def test_model_is_required(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as error:
        _main(
            monkeypatch, tmp_path,
            "--benchmark", "humaneval", "--stage", "pretraining",
            "--hf-repo", "allenai/dolma3_mix-6T-1025-7B",
        )
    assert error.value.code == 2


def test_scanning_the_other_arms_pretraining_mix_is_refused(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as error:
        _main(
            monkeypatch, tmp_path,
            "--benchmark", "humaneval", "--stage", "pretraining",
            "--model", "Olmo3.1-32B-Instruct",
            "--hf-repo", "allenai/dolma3_mix-6T-1025-7B",
        )
    assert error.value.code == 2


def test_a_closed_corpus_arm_is_not_an_allowed_model(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as error:
        _main(
            monkeypatch, tmp_path,
            "--benchmark", "humaneval", "--stage", "sft",
            "--model", "Qwen2.5-7B-Instruct",
            "--hf-repo", "allenai/Dolci-Instruct-SFT",
        )
    assert error.value.code == 2


def test_a_jsonl_scan_keeps_ranked_candidates_with_text_and_both_rules(monkeypatch, tmp_path):
    import json

    from qcd.data.schema import Dataset, Item

    item = Item(item_id="i", dataset=Dataset.HUMANEVAL,
                prompt="one two three four five six seven eight nine ten eleven twelve thirteen")
    monkeypatch.setattr(MODULE, "_benchmark_items", lambda name: [item])
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(row) + "\n" for row in (
        {"id": "verbatim", "text": "x " + item.prompt + " y"},
        {"id": "first-eight", "text": "one two three four five six seven eight"},
        {"id": "unrelated", "text": "nothing in common"},
    )))
    _main(monkeypatch, tmp_path, "--benchmark", "humaneval", "--stage", "sft",
          "--model", "Olmo3-7B-Instruct", "--jsonl", str(corpus))

    rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert [row["document_id"] for row in rows] == ["verbatim", "first-eight"]
    assert [row["candidate_rank"] for row in rows] == [1, 2]
    assert rows[0]["corpus_status"] == rows[0]["sensitivity_corpus_status"] == "confirmed-match"
    # The 8-word document holds one of the prompt's six 8-grams and no 13-gram.
    assert rows[1]["match_detected"] is False
    assert rows[1]["sensitivity_matched_ngrams"] == 1
    assert rows[1]["document_text"] == "one two three four five six seven eight"
    assert {row["coverage_complete"] for row in rows} == {True}
