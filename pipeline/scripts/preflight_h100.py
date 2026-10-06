#!/usr/bin/env python
"""Check that this machine can start the main run, before it starts.

Prints one PASS / FAIL / UNVERIFIED row per check and exits 1 when any row
is FAIL. UNVERIFIED means the check was not run (`--offline` skips every
network check); it does not fail the preflight, so read those rows.

Checks:
  - the checkout is clean at HEAD (`qcd.io.manifest.require_clean_checkout`);
  - Python is 3.12, the H100 image's interpreter (Dockerfile; 3.12.13 per
    pipeline_implementation_log.md), and every `==` pin in
    requirements-h100.txt matches the installed version (a PEP 440 local
    segment such as `+cu130` is ignored; the rest must match exactly);
  - CUDA is available, with the GPU name, total memory and the CUDA version
    torch was built with (informational);
  - the filesystem holding HF_HOME has room for the five bf16 snapshots
    (registry parameter counts x 2 bytes, an estimate: about 173 GB);
  - each registry model's pinned revision resolves on the Hub and this
    account may read it, by fetching `config.json` metadata only (a gated
    repo without access shows here, not at load time);
  - each model's AWQ checkpoint is where `models/loader.py` looks for it, with
    `quantization_manifest.json` and a valid `calibration_overlap_report.json`;
  - each model has a `sample_batch_size` (scripts/measure_sample_batch.py);
  - the pinned LiveCodeBench release_v6 files resolve at the pinned dataset
    revision, again by metadata only.

No weights or dataset files are downloaded.

Usage:
  python scripts/preflight_h100.py
  python scripts/preflight_h100.py --offline
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import math
import platform
import re
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from qcd.config import ModelSpec
from qcd.data.livecodebench import _RELEASE_FILES, _REPO_ID as LCB_REPO_ID, REPO_REVISION as LCB_REPO_REVISION
from qcd.io.manifest import require_calibration_overlap_report, require_clean_checkout
from qcd.models.loader import _quantized_checkpoint_dir
from qcd.models.registry import ALL_MODELS

_PIPELINE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PIPELINE_DIR.parent
_REQUIREMENTS = _PIPELINE_DIR / "requirements-h100.txt"
EXPECTED_PYTHON = "3.12"
LCB_RELEASE = "release_v6"

PASS, FAIL, UNVERIFIED = "PASS", "FAIL", "UNVERIFIED"


@dataclasses.dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


@dataclasses.dataclass
class Environment:
    """Everything the checks read from the machine, so tests can fake it."""

    repo_dir: Path
    offline: bool
    models: tuple[ModelSpec, ...]
    python_version: str
    pins: dict[str, str]
    installed_version: Callable[[str], str | None]
    require_clean_checkout: Callable[[Path], str]
    # (GPU name, total bytes, torch.version.cuda); raises RuntimeError with the reason otherwise.
    cuda_device: Callable[[], tuple[str, int, str | None]]
    hf_home: Path
    free_bytes: Callable[[Path], int]
    min_free_bytes: int
    # (repo_id, filename, revision, repo_type) -> commit the Hub resolved.
    remote_commit: Callable[[str, str, str, str], str]
    awq_checkpoint_dir: Callable[[ModelSpec], Path]


def parse_pins(text: str) -> dict[str, str]:
    pins = {}
    for line in text.splitlines():
        match = re.fullmatch(r"\s*([A-Za-z0-9_.-]+)\s*==\s*(\S+)\s*", line.split("#", 1)[0])
        if match:
            pins[match.group(1)] = match.group(2)
    return pins


def check_git(env: Environment) -> Check:
    try:
        commit = env.require_clean_checkout(env.repo_dir)
    except RuntimeError as error:
        return Check("git checkout", FAIL, str(error))
    return Check("git checkout", PASS, f"clean at {commit}")


def check_python(env: Environment) -> Check:
    if env.python_version.split(".")[:2] == EXPECTED_PYTHON.split("."):
        return Check("python", PASS, env.python_version)
    return Check("python", FAIL, f"{env.python_version}, expected {EXPECTED_PYTHON}.x")


def _public_version(version: str) -> str:
    return version.split("+", 1)[0]


def check_packages(env: Environment) -> list[Check]:
    checks = []
    for package, pinned in env.pins.items():
        installed = env.installed_version(package)
        if installed is not None and _public_version(installed) == _public_version(pinned):
            checks.append(Check(f"package {package}", PASS, installed))
        elif installed is None:
            checks.append(Check(f"package {package}", FAIL, f"not installed, pinned {pinned}"))
        else:
            checks.append(Check(f"package {package}", FAIL, f"installed {installed}, pinned {pinned}"))
    return checks


def check_cuda(env: Environment) -> Check:
    try:
        name, total, cuda_build = env.cuda_device()
    except RuntimeError as error:
        return Check("cuda", FAIL, str(error))
    return Check("cuda", PASS, f"{name}, {total / 1e9:.1f} GB, torch built for CUDA {cuda_build}")


def check_disk(env: Environment) -> Check:
    path = env.hf_home
    while not path.exists():
        path = path.parent
    free = env.free_bytes(path)
    detail = f"{free / 1e9:.0f} GB free at {path} (HF_HOME {env.hf_home}), need {env.min_free_bytes / 1e9:.0f} GB"
    return Check("disk", PASS if free >= env.min_free_bytes else FAIL, detail)


def check_model_revision(env: Environment, spec: ModelSpec) -> Check:
    from huggingface_hub.errors import (  # noqa: PLC0415
        GatedRepoError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError,
    )

    name = f"revision {spec.name}"
    if env.offline:
        return Check(name, UNVERIFIED, "--offline")
    try:
        commit = env.remote_commit(spec.hf_repo_id, "config.json", spec.revision, "model")
    except GatedRepoError:
        return Check(name, FAIL, (
            f"{spec.hf_repo_id} is gated and this account has no access: accept the license "
            "on the Hub and log in (HF_TOKEN or `hf auth login`)"
        ))
    except RevisionNotFoundError:
        return Check(name, FAIL, f"{spec.hf_repo_id} has no revision {spec.revision}")
    except RepositoryNotFoundError:
        return Check(name, FAIL, f"{spec.hf_repo_id} not found")
    except (HfHubHTTPError, OSError) as error:
        return Check(name, FAIL, f"{spec.hf_repo_id}: {type(error).__name__}: {error}")
    if commit != spec.revision:
        return Check(name, FAIL, f"{spec.hf_repo_id}@{spec.revision} resolved to {commit}")
    return Check(name, PASS, f"{spec.hf_repo_id}@{spec.revision}")


def check_awq_checkpoint(env: Environment, spec: ModelSpec) -> Check:
    name = f"awq {spec.name}"
    directory = env.awq_checkpoint_dir(spec)
    if not directory.is_dir():
        return Check(name, FAIL, f"no checkpoint at {directory}")
    if not (directory / "quantization_manifest.json").is_file():
        return Check(name, FAIL, f"{directory} has no quantization_manifest.json")
    try:
        report = require_calibration_overlap_report(directory)
    except (FileNotFoundError, ValueError) as error:
        return Check(name, FAIL, str(error))
    return Check(name, PASS, (
        f"{directory}; calibration overlap {report['n_overlapping_texts']} of "
        f"{report['n_evaluation_texts']} evaluation texts"
    ))


def check_sample_batch_size(spec: ModelSpec) -> Check:
    name = f"sample batch {spec.name}"
    if spec.sample_batch_size is None:
        return Check(name, FAIL, "unset in models/registry.py; measure with scripts/measure_sample_batch.py")
    return Check(name, PASS, str(spec.sample_batch_size))


def check_livecodebench(env: Environment) -> Check:
    from huggingface_hub.errors import HfHubHTTPError  # noqa: PLC0415

    name = f"livecodebench {LCB_RELEASE}"
    if env.offline:
        return Check(name, UNVERIFIED, "--offline")
    files = _RELEASE_FILES[LCB_RELEASE]
    for filename in files:
        try:
            commit = env.remote_commit(LCB_REPO_ID, filename, LCB_REPO_REVISION, "dataset")
        except (HfHubHTTPError, OSError) as error:
            return Check(name, FAIL, f"{filename}: {type(error).__name__}: {error}")
        if commit != LCB_REPO_REVISION:
            return Check(name, FAIL, f"{filename}@{LCB_REPO_REVISION} resolved to {commit}")
    return Check(name, PASS, f"{len(files)} files at {LCB_REPO_ID}@{LCB_REPO_REVISION}")


def run_checks(env: Environment) -> list[Check]:
    return [
        check_git(env),
        check_python(env),
        *check_packages(env),
        check_cuda(env),
        check_disk(env),
        *(check_model_revision(env, spec) for spec in env.models),
        *(check_awq_checkpoint(env, spec) for spec in env.models),
        *(check_sample_batch_size(spec) for spec in env.models),
        check_livecodebench(env),
    ]


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cuda_device() -> tuple[str, int, str | None]:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        raise RuntimeError("torch is not installed") from None
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    properties = torch.cuda.get_device_properties(0)
    return properties.name, properties.total_memory, torch.version.cuda


def _remote_commit(repo_id: str, filename: str, revision: str, repo_type: str) -> str:
    from huggingface_hub import get_hf_file_metadata, hf_hub_url  # noqa: PLC0415

    url = hf_hub_url(repo_id, filename, repo_type=repo_type, revision=revision)
    return get_hf_file_metadata(url).commit_hash


def _bf16_snapshot_bytes(models: tuple[ModelSpec, ...]) -> int:
    return math.ceil(sum(spec.param_count_b for spec in models) * 2) * 10**9


def real_environment(*, offline: bool, min_free_gb: float | None) -> Environment:
    from huggingface_hub import constants  # noqa: PLC0415

    return Environment(
        repo_dir=_REPO_ROOT,
        offline=offline,
        models=ALL_MODELS,
        python_version=platform.python_version(),
        pins=parse_pins(_REQUIREMENTS.read_text(encoding="utf-8")),
        installed_version=_installed_version,
        require_clean_checkout=require_clean_checkout,
        cuda_device=_cuda_device,
        hf_home=Path(constants.HF_HOME),
        free_bytes=lambda path: shutil.disk_usage(path).free,
        min_free_bytes=(
            _bf16_snapshot_bytes(ALL_MODELS) if min_free_gb is None else int(min_free_gb * 10**9)
        ),
        remote_commit=_remote_commit,
        awq_checkpoint_dir=_quantized_checkpoint_dir,
    )


def print_table(checks: list[Check]) -> None:
    width = max(len(check.name) for check in checks)
    for check in checks:
        print(f"{check.status:<10}  {check.name:<{width}}  {check.detail}")
    counts = {status: sum(c.status == status for c in checks) for status in (PASS, FAIL, UNVERIFIED)}
    print(f"\n{counts[PASS]} PASS, {counts[FAIL]} FAIL, {counts[UNVERIFIED]} UNVERIFIED")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="Skip the Hub checks; they report UNVERIFIED.")
    parser.add_argument("--min-free-gb", type=float, default=None,
                        help="Free space needed at HF_HOME. Default: the five bf16 snapshots, ~173 GB.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    checks = run_checks(real_environment(offline=args.offline, min_free_gb=args.min_free_gb))
    print_table(checks)
    return 1 if any(check.status == FAIL for check in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
