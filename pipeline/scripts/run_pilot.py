#!/usr/bin/env python
"""H100 pilot driver (paper §4.7): Qwen2.5-7B and Olmo3-7B, BNB-nf4 arm
first. Thin CLI wrapper over qcd.real_run.run() — see that module's
docstring for what does and doesn't work yet: everything up through model
loading is real and tested; `load_model(..., mock=False)` still raises
NotImplementedError (GPU paths deferred to a future session, per
pipeline_build_plan.md and the scope agreed for this pipeline-construction
pass).

Usage: python scripts/run_pilot.py [--output-dir data/raw/pilot]
                                    [--lcb-cutoff 2024-09-01] [--item-limit 50]
"""

import argparse
import datetime as dt
from pathlib import Path

from qcd.config import Quant
from qcd.constants import CDD_N_SAMPLES
from qcd.models.registry import PILOT_MODELS
from qcd.real_run import RealRunConfig, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/pilot"))
    parser.add_argument(
        "--lcb-cutoff", type=str, default="2024-09-01",
        help="Conservative per-arm boundary per §4.2's evidence-tier rule "
             "(provisional Qwen2.5 2024-09 release date, pending §5 step 4 verification).",
    )
    parser.add_argument("--lcb-release", type=str, default="release_v6")
    parser.add_argument(
        "--item-limit", type=int, default=50,
        help="Pilot-scale per-condition item cap. Pass -1 to disable (use every item).",
    )
    parser.add_argument("--n-cdd-samples", type=int, default=CDD_N_SAMPLES)
    args = parser.parse_args()

    config = RealRunConfig(
        models=PILOT_MODELS,
        quant_levels=(Quant.FP16, Quant.BNB_NF4),
        output_dir=args.output_dir,
        lcb_cutoff_boundary=dt.datetime.fromisoformat(args.lcb_cutoff),
        lcb_release_version=args.lcb_release,
        n_cdd_samples=args.n_cdd_samples,
        item_limit_per_condition=None if args.item_limit < 0 else args.item_limit,
    )
    run(config)


if __name__ == "__main__":
    main()
