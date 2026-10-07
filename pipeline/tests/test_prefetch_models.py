"""`scripts/prefetch_models.py`: every pinned download is requested at its
pinned revision, concurrently, and a failure makes the script fail."""

import importlib.util
import threading
from pathlib import Path

import pytest

from qcd.data.livecodebench import REPO_REVISION as LCB_REPO_REVISION
from qcd.models.registry import MAIN_ANALYSIS_MODELS

_SCRIPT = Path(__file__).parents[1] / "scripts" / "prefetch_models.py"
_SPEC = importlib.util.spec_from_file_location("prefetch_models", _SCRIPT)
PREFETCH = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(PREFETCH)


def test_every_model_and_dataset_is_fetched_at_its_pinned_revision():
    calls = []

    def download(repo_id, *, revision, repo_type):
        calls.append((repo_id, revision, repo_type))
        return f"/cache/{repo_id}"

    assert PREFETCH.prefetch(download) == []
    expected = {(m.hf_repo_id, m.revision, "model") for m in MAIN_ANALYSIS_MODELS}
    assert expected <= set(calls)
    assert ("livecodebench/code_generation_lite", LCB_REPO_REVISION, "dataset") in calls
    assert len(calls) == len(MAIN_ANALYSIS_MODELS) + 2


def test_downloads_run_concurrently():
    n = len(PREFETCH.snapshots())
    barrier = threading.Barrier(n, timeout=5)

    def download(repo_id, *, revision, repo_type):
        barrier.wait()  # raises BrokenBarrierError unless all n run at once
        return repo_id

    assert PREFETCH.prefetch(download) == []


def test_a_failed_download_is_reported_and_fails_the_script(monkeypatch):
    def download(repo_id, *, revision, repo_type):
        if repo_id.startswith("meta-llama/"):
            raise PermissionError("gated")
        return repo_id

    failures = PREFETCH.prefetch(download)
    assert [repo for repo, _ in failures] == ["meta-llama/Llama-3.1-8B-Instruct"]

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    with pytest.raises(SystemExit) as exit_info:
        PREFETCH.main([])
    assert exit_info.value.code == 1


def test_help_prints_usage_without_downloading(monkeypatch, capsys):
    import huggingface_hub

    def download(*args, **kwargs):
        raise AssertionError("--help must not download")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    with pytest.raises(SystemExit) as exit_info:
        PREFETCH.main(["--help"])
    assert exit_info.value.code == 0
    assert "pinned revision" in capsys.readouterr().out
