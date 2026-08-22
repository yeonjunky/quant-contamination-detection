"""Per-run manifest — git commit, config hash, installed-package versions,
seeds, timestamps (pipeline_build_plan.md's raw-data-schema section): the
record that lets a later re-analysis (or the boundary-sensitivity re-runs
of paper §4.2, or a reviewer) know exactly what produced a given
`data/raw/` tree.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import platform
import os
import subprocess
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DEFAULT_TRACKED_PACKAGES: tuple[str, ...] = (
    "numpy", "pandas", "pyarrow", "transformers", "torch", "bitsandbytes",
    "accelerate", "gptqmodel", "llmcompressor", "evalplus", "datasets",
    "huggingface_hub", "statsmodels",
)


def get_git_commit_hash(repo_dir: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def get_installed_package_versions(packages: tuple[str, ...] = DEFAULT_TRACKED_PACKAGES) -> dict[str, str | None]:
    """Reads installed-distribution metadata (no import needed), so this is
    safe and cheap to call even for GPU packages not installed on the
    mock-only profile — they simply resolve to None."""
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def config_hash(config: dict) -> str:
    """Stable hash of a JSON-serializable config dict — `sort_keys=True` so
    key order never changes the hash, `default=str` so non-JSON-native
    values (enums, Paths) don't crash it."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclasses.dataclass
class RunManifest:
    git_commit: str | None
    config: dict
    config_hash: str
    package_versions: dict[str, str | None]
    seed: int | None
    timestamp_utc: str
    hostname: str
    platform: str
    python_version: str
    extra: dict = dataclasses.field(default_factory=dict)


def build_manifest(
    config: dict,
    *,
    packages: tuple[str, ...] = DEFAULT_TRACKED_PACKAGES,
    seed: int | None = None,
    repo_dir: str | Path | None = None,
    extra: dict | None = None,
) -> RunManifest:
    return RunManifest(
        git_commit=get_git_commit_hash(repo_dir),
        config=dict(config),
        config_hash=config_hash(config),
        package_versions=get_installed_package_versions(packages),
        seed=seed,
        timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        hostname=platform.node(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        extra=extra or {},
    )


def write_manifest(manifest: RunManifest, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(manifest), f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path


def read_manifest(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)
