"""`scripts/preflight_h100.py`: each check against a faked machine, no
network, torch or GPU."""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError

from qcd.data.livecodebench import REPO_REVISION as LCB_REPO_REVISION
from qcd.models.loader import _quantized_checkpoint_dir
from qcd.models.registry import ALL_MODELS, LLAMA3_1_8B, QWEN2_5_7B, QWEN2_5_32B

_SCRIPT = Path(__file__).parents[1] / "scripts" / "preflight_h100.py"
_SPEC = importlib.util.spec_from_file_location("preflight_h100", _SCRIPT)
PRE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = PRE
_SPEC.loader.exec_module(PRE)

GB = 10**9


def _write_awq(directory: Path, *, manifest=True, report=True):
    directory.mkdir(parents=True)
    if manifest:
        (directory / "quantization_manifest.json").write_text("{}")
    if report:
        (directory / "calibration_overlap_report.json").write_text(json.dumps(
            {"method": "13-gram", "n_evaluation_texts": 1597, "n_overlapping_texts": 0}
        ))


def _env(tmp_path, **overrides) -> "PRE.Environment":
    for spec in ALL_MODELS:
        if not (tmp_path / spec.name).exists():
            _write_awq(tmp_path / spec.name)
    env = PRE.Environment(
        repo_dir=tmp_path,
        offline=False,
        models=(QWEN2_5_7B,),
        python_version="3.12.13",
        pins={"torch": "2.13.0"},
        installed_version={"torch": "2.13.0"}.get,
        require_clean_checkout=lambda repo: "a" * 40,
        cuda_device=lambda: ("NVIDIA H100 80GB HBM3", 80 * GB, "13.0"),
        hf_home=tmp_path,
        free_bytes=lambda path: 500 * GB,
        min_free_bytes=173 * GB,
        remote_commit=lambda repo_id, filename, revision, repo_type: revision,
        awq_checkpoint_dir=lambda spec: tmp_path / spec.name,
    )
    return dataclasses.replace(env, **overrides)


def _raise(error):
    def call(*args, **kwargs):
        raise error
    return call


def test_a_fully_ready_machine_has_no_failures(tmp_path):
    checks = PRE.run_checks(_env(tmp_path))
    assert {check.status for check in checks} == {PRE.PASS}


def test_git_check(tmp_path):
    assert PRE.check_git(_env(tmp_path)).status == PRE.PASS
    dirty = _env(tmp_path, require_clean_checkout=_raise(RuntimeError("tracked files have uncommitted changes")))
    result = PRE.check_git(dirty)
    assert result.status == PRE.FAIL and "uncommitted" in result.detail


def test_python_check(tmp_path):
    assert PRE.check_python(_env(tmp_path)).status == PRE.PASS
    assert PRE.check_python(_env(tmp_path, python_version="3.14.6")).status == PRE.FAIL


def test_package_checks(tmp_path):
    env = _env(
        tmp_path,
        pins={"torch": "2.13.0", "numpy": "2.4.6", "bitsandbytes": "0.50.1"},
        installed_version={"torch": "2.13.0", "numpy": "2.5.1"}.get,
    )
    by_name = {check.name: check for check in PRE.check_packages(env)}
    assert by_name["package torch"].status == PRE.PASS
    assert by_name["package numpy"].status == PRE.FAIL
    assert by_name["package numpy"].detail == "installed 2.5.1, pinned 2.4.6"
    assert by_name["package bitsandbytes"].status == PRE.FAIL
    assert "not installed" in by_name["package bitsandbytes"].detail


@pytest.mark.parametrize(("pinned", "installed", "status"), [
    ("2.13.0", "2.13.0+cu130", PRE.PASS),
    ("2.13.0+cu130", "2.13.0", PRE.PASS),
    ("2.13.0+cu130", "2.13.0+cu124", PRE.PASS),
    ("2.13.0", "2.13.1+cu130", PRE.FAIL),
    ("2.13.0", "2.13.0.post1", PRE.FAIL),
    ("2.13.0", "2.13", PRE.FAIL),
])
def test_package_check_ignores_only_the_local_version_segment(tmp_path, pinned, installed, status):
    env = _env(tmp_path, pins={"torch": pinned}, installed_version={"torch": installed}.get)
    [check] = PRE.check_packages(env)
    assert check.status == status


def test_every_h100_pin_is_the_version_in_the_recorded_h100_freeze():
    def normalized(pins):
        return {name.lower().replace("_", "-"): version for name, version in pins.items()}

    pipeline = Path(__file__).parents[1]
    pins = normalized(PRE.parse_pins((pipeline / "requirements-h100.txt").read_text(encoding="utf-8")))
    freeze = normalized(PRE.parse_pins((pipeline / "envs" / "local-smoke-freeze.txt").read_text(encoding="utf-8")))
    assert {name: freeze.get(name) for name in pins} == pins


def test_parse_pins_keeps_only_exact_pins():
    text = "# torch==0.0\ntorch==2.13.0+cu130  # note\nzstandard\nnumpy == 2.4.6\n"
    assert PRE.parse_pins(text) == {"torch": "2.13.0+cu130", "numpy": "2.4.6"}


def test_cuda_check(tmp_path):
    result = PRE.check_cuda(_env(tmp_path))
    assert result.status == PRE.PASS
    assert result.detail == "NVIDIA H100 80GB HBM3, 80.0 GB, torch built for CUDA 13.0"
    missing = PRE.check_cuda(_env(tmp_path, cuda_device=_raise(RuntimeError("torch is not installed"))))
    assert missing.status == PRE.FAIL and missing.detail == "torch is not installed"


def test_disk_check_measures_the_nearest_existing_directory(tmp_path):
    seen = []
    env = _env(tmp_path, hf_home=tmp_path / "not" / "created", free_bytes=lambda p: seen.append(p) or 500 * GB)
    assert PRE.check_disk(env).status == PRE.PASS
    assert seen == [tmp_path]
    assert PRE.check_disk(_env(tmp_path, free_bytes=lambda p: 79 * GB)).status == PRE.FAIL


def test_model_revision_check(tmp_path):
    assert PRE.check_model_revision(_env(tmp_path), QWEN2_5_7B).status == PRE.PASS
    offline = PRE.check_model_revision(_env(tmp_path, offline=True), QWEN2_5_7B)
    assert offline.status == PRE.UNVERIFIED

    moved = _env(tmp_path, remote_commit=lambda *args: "b" * 40)
    assert PRE.check_model_revision(moved, QWEN2_5_7B).status == PRE.FAIL

    response = httpx.Response(403, request=httpx.Request("HEAD", "https://huggingface.co/x"))
    gated = _env(tmp_path, remote_commit=_raise(GatedRepoError("gated", response=response)))
    result = PRE.check_model_revision(gated, LLAMA3_1_8B)
    assert result.status == PRE.FAIL
    assert "gated" in result.detail and "meta-llama/Llama-3.1-8B-Instruct" in result.detail


def test_model_revision_check_asks_for_the_pinned_revision(tmp_path):
    calls = []
    env = _env(tmp_path, remote_commit=lambda *args: calls.append(args) or args[2])
    PRE.check_model_revision(env, QWEN2_5_32B)
    assert calls == [(QWEN2_5_32B.hf_repo_id, "config.json", QWEN2_5_32B.revision, "model")]


def test_awq_checkpoint_check(tmp_path):
    assert PRE.check_awq_checkpoint(_env(tmp_path), QWEN2_5_7B).status == PRE.PASS

    missing = _env(tmp_path, awq_checkpoint_dir=lambda spec: tmp_path / "absent")
    assert PRE.check_awq_checkpoint(missing, QWEN2_5_7B).status == PRE.FAIL

    _write_awq(tmp_path / "no-manifest", manifest=False)
    no_manifest = _env(tmp_path, awq_checkpoint_dir=lambda spec: tmp_path / "no-manifest")
    result = PRE.check_awq_checkpoint(no_manifest, QWEN2_5_7B)
    assert result.status == PRE.FAIL and "quantization_manifest.json" in result.detail

    _write_awq(tmp_path / "no-report", report=False)
    no_report = _env(tmp_path, awq_checkpoint_dir=lambda spec: tmp_path / "no-report")
    result = PRE.check_awq_checkpoint(no_report, QWEN2_5_7B)
    assert result.status == PRE.FAIL and "calibration_overlap_report.json" in result.detail


def test_sample_batch_size_check():
    assert PRE.check_sample_batch_size(QWEN2_5_7B).status == PRE.PASS
    unset = PRE.check_sample_batch_size(dataclasses.replace(QWEN2_5_7B, sample_batch_size=None))
    assert unset.status == PRE.FAIL


def test_livecodebench_check(tmp_path):
    calls = []
    env = _env(tmp_path, remote_commit=lambda *args: calls.append(args) or args[2])
    assert PRE.check_livecodebench(env).status == PRE.PASS
    assert [call[1] for call in calls] == [
        "test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl",
    ]
    assert {(call[0], call[2], call[3]) for call in calls} == {
        ("livecodebench/code_generation_lite", LCB_REPO_REVISION, "dataset")
    }

    assert PRE.check_livecodebench(_env(tmp_path, offline=True)).status == PRE.UNVERIFIED
    broken = _env(tmp_path, remote_commit=_raise(OSError("connection refused")))
    result = PRE.check_livecodebench(broken)
    assert result.status == PRE.FAIL and "connection refused" in result.detail


def test_main_exits_nonzero_on_any_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(PRE, "real_environment", lambda **kwargs: _env(tmp_path))
    assert PRE.main([]) == 0
    monkeypatch.setattr(PRE, "real_environment", lambda **kwargs: _env(tmp_path, python_version="3.10.0"))
    assert PRE.main([]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_unverified_alone_does_not_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(PRE, "real_environment", lambda **kwargs: _env(tmp_path, offline=True))
    assert PRE.main(["--offline"]) == 0


def test_real_environment_reuses_the_loader_path_and_registry():
    env = PRE.real_environment(offline=True, min_free_gb=None)
    assert env.awq_checkpoint_dir is _quantized_checkpoint_dir
    assert env.models == ALL_MODELS
    # 2 bytes x (7 + 32.5 + 8 + 7 + 32) billion parameters.
    assert env.min_free_bytes == 173 * GB
