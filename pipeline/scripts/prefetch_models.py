#!/usr/bin/env python
"""Download every registry model snapshot at its pinned revision, plus the
pinned LiveCodeBench and AWQ-calibration dataset files, all at once.

Downloads are network-bound and use no GPU, so they run concurrently with
each other and can run in the background while other H100 steps proceed.
Later loads (`quantize_model.py`, the smoke tests, the main run) then read
the local Hugging Face cache instead of downloading one model at a time.

Usage:
  python scripts/prefetch_models.py
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qcd.data.livecodebench import _REPO_ID as LCB_REPO_ID, REPO_REVISION as LCB_REPO_REVISION  # noqa: E402
from qcd.models.registry import MAIN_ANALYSIS_MODELS  # noqa: E402
from quantize_model import CALIBRATION_DATASET_ID, CALIBRATION_DATASET_REVISION  # noqa: E402


def snapshots() -> list[tuple[str, str, str]]:
    """(repo_id, revision, repo_type) for every pinned download."""
    return [
        *((model.hf_repo_id, model.revision, "model") for model in MAIN_ANALYSIS_MODELS),
        (LCB_REPO_ID, LCB_REPO_REVISION, "dataset"),
        (CALIBRATION_DATASET_ID, CALIBRATION_DATASET_REVISION, "dataset"),
    ]


def prefetch(download) -> list[tuple[str, str]]:
    """Run every download concurrently; return (repo_id, error) for failures."""
    failures = []
    with ThreadPoolExecutor(max_workers=len(snapshots())) as pool:
        futures = {
            pool.submit(download, repo_id, revision=revision, repo_type=repo_type): repo_id
            for repo_id, revision, repo_type in snapshots()
        }
        for future, repo_id in futures.items():
            try:
                print(f"done  {repo_id}: {future.result()}")
            except Exception as error:  # noqa: BLE001 - report every failure, then exit non-zero
                print(f"FAIL  {repo_id}: {error}")
                failures.append((repo_id, str(error)))
    return failures


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args(argv)
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    sys.exit(1 if prefetch(snapshot_download) else 0)


if __name__ == "__main__":
    main()
