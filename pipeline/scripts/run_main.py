#!/usr/bin/env python
"""H100 main-experiment driver (paper §5 step 8): all five
MAIN_ANALYSIS_MODELS, the full quantization ladder. Thin CLI wrapper over
qcd.real_run.run(); the real bf16/bnb/AWQ loading and scoring paths are
implemented. Complete engineering validation and freeze the operational
configuration before invoking this study-data driver. Validation outputs must
not be used to change C1-C4, sample planning, or detector priority.

Writes the run manifest with `study_phase="main_study"` (paper §4.6), which is
what the analysis side requires before it will read a raw tree. Refuses to
start from a checkout with uncommitted tracked changes, so the manifest's
commit is the code that produced the data.

Every setting is frozen in code; the only choice left to the command line is
which (model, precision) cells this process runs. Run one cell per process:
each writes into the same `data/raw/main` under the full study's manifest, a
completed cell is skipped without loading its model, and an interrupted cell
resumes from its last written part.

Usage:
    python scripts/run_main.py                                   # every cell
    python scripts/run_main.py --cell Qwen2.5-7B-Instruct:bnb_nf4
"""

import argparse
import datetime as dt
from pathlib import Path

from qcd.config import Quant
from qcd.constants import LCB_SHARED_CONTROL_BOUNDARY
from qcd.io.manifest import StudyPhase, require_clean_checkout
from qcd.models.registry import MAIN_ANALYSIS_MODELS
from qcd.real_run import RealRunConfig, run

MAIN_QUANT_LEVELS = (Quant.BF16, Quant.BNB_INT8, Quant.BNB_NF4, Quant.GPTQ_AWQ_INT4)

# Repo-root-anchored, not CWD-relative: a run started from `pipeline/` used to
# land in `pipeline/data/raw/main`, which `.gitignore`'s root-anchored `/data/`
# does not cover and `scripts/sync_from_h100.sh` does not fetch. Matches the
# `data/raw/{validation,main}` layout in pipeline_build_plan.md.
MAIN_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "main"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cell", action="append", metavar="MODEL:QUANT",
        choices=[f"{m.name}:{q.value}" for m in MAIN_ANALYSIS_MODELS for q in MAIN_QUANT_LEVELS],
        help="run only this (model, precision) cell; repeatable. Default: every cell.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_clean_checkout()

    config = RealRunConfig(
        models=MAIN_ANALYSIS_MODELS,
        quant_levels=MAIN_QUANT_LEVELS,
        output_dir=MAIN_OUTPUT_DIR,
        lcb_cutoff_boundary=dt.datetime.fromisoformat(LCB_SHARED_CONTROL_BOUNDARY),
        item_limit_per_condition=None,
        study_phase=StudyPhase.MAIN_STUDY,
        cells=(
            None if args.cell is None
            else frozenset(
                (name, Quant(quant)) for name, quant in (c.rsplit(":", 1) for c in args.cell)
            )
        ),
    )
    run(config)


if __name__ == "__main__":
    main()
