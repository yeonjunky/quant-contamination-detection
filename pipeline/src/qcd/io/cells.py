"""The study's (model, precision) cells on disk.

`real_run.py` runs one cell per process and writes `cells/<cell>/complete.json`
after the cell's last raw part. The analysis reads a tree only when every
cell the manifest names carries that record, so a partly finished study is
refused instead of silently analysed with the missing items dropped.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from qcd.io.raw_writer import part_is_written


def cell_id(model_name: str, quant: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{model_name}-{quant}")


def completion_marker_path(run_dir: str | Path, cell: str) -> Path:
    return Path(run_dir) / "cells" / cell / "complete.json"


def require_complete_study(run_dir: str | Path, manifest: dict, *, consumer: str = "analysis") -> None:
    """Refuse unless every model x precision in the manifest config has a
    completion record carrying the manifest's `config_hash` whose parts all
    exist under `raw/`."""
    run_dir = Path(run_dir)
    config = manifest.get("config", {})
    if "models" not in config or "quant_levels" not in config:
        raise ValueError(
            f"{consumer} refused: the manifest in {run_dir} names no `models` / `quant_levels`, "
            "so the study's cells cannot be checked for completion."
        )
    problems = []
    for model in config["models"]:
        for quant in config["quant_levels"]:
            cell = cell_id(model, quant)
            marker = completion_marker_path(run_dir, cell)
            if not marker.exists():
                problems.append(f"{cell}: no completion record")
                continue
            record = json.loads(marker.read_text(encoding="utf-8"))
            if record.get("config_hash") != manifest.get("config_hash"):
                problems.append(f"{cell}: completion record has a different config_hash")
                continue
            missing = [p for p in record.get("parts", []) if not part_is_written(run_dir / "raw", p)]
            if not record.get("parts") or missing:
                problems.append(f"{cell}: parts missing on disk {missing or '(none listed)'}")
    if problems:
        raise ValueError(
            f"{consumer} refused: {run_dir} is not a complete study. Incomplete cells:\n  "
            + "\n  ".join(problems)
        )
