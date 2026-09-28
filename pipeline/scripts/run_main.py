#!/usr/bin/env python
"""H100 main-experiment driver (paper §5 step 8): all five
MAIN_ANALYSIS_MODELS, the full quantization ladder. Thin CLI wrapper over
qcd.real_run.run(); the real bf16/bnb/AWQ loading and scoring paths are
implemented. Complete engineering validation and freeze the operational
configuration before invoking this study-data driver. Validation outputs must
not be used to change C1-C4, sample planning, or detector priority.

Writes the run manifest with `study_phase="main_study"` (paper §4.6), which is
what the analysis side requires before it will read a raw tree.

Usage: python scripts/run_main.py
"""

import argparse
import datetime as dt
from pathlib import Path

from qcd.config import Quant
from qcd.constants import CDD_N_SAMPLES, LCB_SHARED_CONTROL_BOUNDARY
from qcd.io.manifest import StudyPhase
from qcd.models.registry import MAIN_ANALYSIS_MODELS
from qcd.real_run import RealRunConfig, run

# Repo-root-anchored, not CWD-relative: a run started from `pipeline/` used to
# land in `pipeline/data/raw/main`, which `.gitignore`'s root-anchored `/data/`
# does not cover and `scripts/sync_from_h100.sh` does not fetch. Matches the
# `data/raw/{validation,main}` layout in pipeline_build_plan.md.
MAIN_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "main"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=MAIN_OUTPUT_DIR)
    parser.add_argument(
        "--lcb-cutoff", type=str, default=LCB_SHARED_CONTROL_BOUNDARY,
        help="Common shared-control boundary (paper §4.2/§5 step 3, fixed at "
             "%(default)s = constants.LCB_SHARED_CONTROL_BOUNDARY). The override "
             "exists for §4.2's boundary-sensitivity re-runs; the main run uses the "
             "fixed value.",
    )
    parser.add_argument("--lcb-release", type=str, default="release_v6")
    parser.add_argument("--n-cdd-samples", type=int, default=CDD_N_SAMPLES)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = RealRunConfig(
        models=MAIN_ANALYSIS_MODELS,
        quant_levels=(Quant.BF16, Quant.BNB_INT8, Quant.BNB_NF4, Quant.GPTQ_AWQ_INT4),
        output_dir=args.output_dir,
        lcb_cutoff_boundary=dt.datetime.fromisoformat(args.lcb_cutoff),
        lcb_release_version=args.lcb_release,
        n_cdd_samples=args.n_cdd_samples,
        item_limit_per_condition=None,
        study_phase=StudyPhase.MAIN_STUDY,
    )
    run(config)


if __name__ == "__main__":
    main()
