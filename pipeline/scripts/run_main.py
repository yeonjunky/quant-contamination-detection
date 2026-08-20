#!/usr/bin/env python
"""H100 main-experiment driver (paper §5 step 8): all five
MAIN_ANALYSIS_MODELS, the full quantization ladder. Thin CLI wrapper over
qcd.real_run.run(); the real bf16/bnb/AWQ loading and scoring paths are
implemented. Run the pre-pilot audit and pilot gate before this full driver.

Usage: python scripts/run_main.py --lcb-cutoff <final-common-boundary>
"""

import argparse
import datetime as dt
from pathlib import Path

from qcd.config import Quant
from qcd.constants import CDD_N_SAMPLES
from qcd.models.registry import MAIN_ANALYSIS_MODELS
from qcd.real_run import RealRunConfig, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/main"))
    parser.add_argument(
        "--lcb-cutoff", type=str, required=True,
        help="Conservative post-cutoff boundary per §4.2's evidence-tier rule — "
             "the latest of the per-arm bounds (must be supplied explicitly for the "
             "main run, not silently defaulted, since it directly determines the "
             "contamination labels the analysis depends on).",
    )
    parser.add_argument("--lcb-release", type=str, default="release_v6")
    parser.add_argument("--n-cdd-samples", type=int, default=CDD_N_SAMPLES)
    args = parser.parse_args()

    config = RealRunConfig(
        models=MAIN_ANALYSIS_MODELS,
        quant_levels=(Quant.BF16, Quant.BNB_INT8, Quant.BNB_NF4, Quant.GPTQ_AWQ_INT4),
        output_dir=args.output_dir,
        lcb_cutoff_boundary=dt.datetime.fromisoformat(args.lcb_cutoff),
        lcb_release_version=args.lcb_release,
        n_cdd_samples=args.n_cdd_samples,
        item_limit_per_condition=None,
    )
    run(config)


if __name__ == "__main__":
    main()
