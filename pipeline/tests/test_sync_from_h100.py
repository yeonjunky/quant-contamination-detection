"""`scripts/sync_from_h100.sh` must not merge paper §4.6's two namespaces.

The script used to pull `data/raw/` whole, which fetched engineering-validation
output and main-study data into one local tree. These tests run it against a
stub `rsync` (no network, no ssh) inside a copy of the repository layout, and
check which remote paths it asks for and what it reports about `study_phase`.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "sync_from_h100.sh"


@pytest.fixture
def sandbox(tmp_path):
    """A copy of the repo's `pipeline/scripts/` layout plus a stub rsync.

    The script derives its local destination from its own location, so copying
    it into `<tmp>/pipeline/scripts/` keeps every write inside `tmp_path`.
    """
    scripts = tmp_path / "pipeline" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(SCRIPT, scripts / SCRIPT.name)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "rsync.log"
    stub = bin_dir / "rsync"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> {log}\n'
    )
    stub.chmod(0o755)
    return tmp_path, scripts / SCRIPT.name, log


def _run(sandbox, *args):
    tmp_path, script, log = sandbox
    environment = dict(os.environ, PATH=f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    result = subprocess.run(
        ["bash", str(script), *args], capture_output=True, text=True, env=environment, check=True,
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls


def test_default_sync_fetches_only_the_main_study_namespace(sandbox):
    result, calls = _run(sandbox, "h100-box")

    assert len(calls) == 1
    assert "/data/raw/main/" in calls[0]
    assert "/data/raw/validation/" not in calls[0]
    # The old behaviour: one rsync of the whole `data/raw/` tree.
    assert "h100-box:~/quant-contamination-detection/data/raw/ " not in calls[0]
    assert "data/raw/main" in result.stdout


def test_validation_namespace_is_opt_in_and_lands_in_its_own_directory(sandbox):
    tmp_path, _, _ = sandbox
    _, calls = _run(sandbox, "--with-validation", "h100-box")

    assert len(calls) == 2
    assert "/data/raw/main/" in calls[0]
    assert "/data/raw/validation/" in calls[1]
    assert (tmp_path / "data" / "raw" / "main").is_dir()
    assert (tmp_path / "data" / "raw" / "validation").is_dir()
    assert str(tmp_path / "data" / "raw" / "main") in calls[0]
    assert str(tmp_path / "data" / "raw" / "validation") in calls[1]


def test_validation_only_skips_the_main_namespace(sandbox):
    _, calls = _run(sandbox, "--validation-only", "h100-box")

    assert len(calls) == 1
    assert "/data/raw/validation/" in calls[0]


def test_extra_rsync_args_reach_rsync_and_not_the_namespace_argument(sandbox):
    _, calls = _run(sandbox, "h100-box", "~/repo", "--", "--dry-run")

    assert len(calls) == 1
    assert calls[0].endswith("--dry-run")
    assert "~/repo/data/raw/main/" in calls[0]
    assert " main " not in calls[0]


def test_it_reports_the_study_phase_recorded_in_each_synced_run(sandbox):
    tmp_path, _, _ = sandbox
    run_dir = tmp_path / "data" / "raw" / "main" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"study_phase": "main_study", "config": {}}, indent=2)
    )

    result, _ = _run(sandbox, "h100-box")

    assert "study_phase=main_study" in result.stdout
    assert str(run_dir) in result.stdout


def test_a_tree_without_a_manifest_is_reported_rather_than_assumed(sandbox):
    result, _ = _run(sandbox, "h100-box")

    assert "no manifest.json" in result.stdout
