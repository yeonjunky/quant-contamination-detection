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
