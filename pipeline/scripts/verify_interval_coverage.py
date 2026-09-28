#!/usr/bin/env python
"""Run paper §4.5.5's pre-main-run interval-coverage check on synthetic data.

§4.5.5: "Verify interval coverage on synthetic data before the main run.
Generate data from the design's own shape — one model's item counts (e.g. 690
`possible-exposure` and 182 `shared-clean-control`), a normal item-difficulty
distribution, and a known β_QE — and estimate the achieved coverage of each
candidate interval method over enough replications to separate it from the
nominal 95%. Record the generating parameters, the number of replications,
and the achieved coverage of every method examined in the analysis manifest."

This script writes exactly that record. It is CPU-only, needs no GPU, no
network and no run directory, and touches no study data. Its output fixes
which method supplies the reported β_QE interval; it is not a study result.

Usage:
    python scripts/verify_interval_coverage.py --replications 200

The record is written to `pipeline/analysis_artifacts/interval_coverage_check.json`
by default — a tracked path, not a temp directory, because `scripts/run_analysis.py`
reads it back and copies it into the analysis manifest (§4.5.5). Pass `--out -`
to print it instead.

By default it runs two scenarios — true β_QE = 0 and a non-zero value —
because a method can hit nominal coverage at one and miss at the other.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:  # allow running without an editable install
    sys.path.insert(0, str(REPO_SRC))

from qcd.analysis.coverage import (  # noqa: E402
    INTERVAL_COVERAGE_RECORD_ID,
    INTERVAL_COVERAGE_RECORD_PATH,
    INTERVAL_METHODS,
    SyntheticDesign,
    estimate_interval_coverage,
)


def _library_versions() -> dict[str, str]:
    import numpy
    import pandas
    import scipy
    import statsmodels

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "statsmodels": statsmodels.__version__,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replications",
        type=int,
        default=200,
        help="replications per scenario (§4.5.5: enough to separate achieved from nominal 95%%)",
    )
    parser.add_argument(
        "--beta-qe",
        type=float,
        nargs="+",
        default=[0.0, 0.5],
        help="true β_QE values to run, one scenario each",
    )
    parser.add_argument("--n-possible-exposure", type=int, default=690)
    parser.add_argument("--n-shared-clean-control", type=int, default=182)
    parser.add_argument("--seed", type=int, default=20250918)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=sorted(INTERVAL_METHODS),
        choices=sorted(INTERVAL_METHODS),
        help="candidate interval methods to check",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(INTERVAL_COVERAGE_RECORD_PATH),
        help="write the JSON record here; '-' prints it instead (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    methods = {name: INTERVAL_METHODS[name] for name in args.methods}
    scenarios = []
    started = time.time()
    for beta_qe in args.beta_qe:
        design = SyntheticDesign(
            n_possible_exposure=args.n_possible_exposure,
            n_shared_clean_control=args.n_shared_clean_control,
            beta_qe=beta_qe,
        )
        scenario_started = time.time()
        results = estimate_interval_coverage(
            design=design,
            n_replications=args.replications,
            methods=methods,
            seed=args.seed,
        )
        scenarios.append(
            {
                "design": design.as_dict(),
                "n_replications": args.replications,
                "seed": args.seed,
                "elapsed_seconds": time.time() - scenario_started,
                "coverage": {name: result.as_dict() for name, result in results.items()},
            }
        )
        for name, result in results.items():
            coverage = "n/a" if result.coverage is None else f"{result.coverage:.3f}"
            se = (
                "n/a"
                if result.coverage_monte_carlo_se is None
                else f"{result.coverage_monte_carlo_se:.3f}"
            )
            width = "n/a" if result.mean_interval_width is None else f"{result.mean_interval_width:.3f}"
            print(
                f"beta_qe={beta_qe:+.2f}  {name:<32} coverage={coverage} (MC SE {se})  "
                f"mean width={width}  estimable={result.n_estimable}/{args.replications}"
            )

    record = {
        "record": INTERVAL_COVERAGE_RECORD_ID,
        "nominal_level": 0.95,
        "adopted_interval_method": "conditional_logit_wald",
        "methods_examined": sorted(methods),
        "scenarios": scenarios,
        "total_elapsed_seconds": time.time() - started,
        "library_versions": _library_versions(),
    }
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
    if args.out == "-":
        print(payload)
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"wrote {out_path}")
    print(f"total elapsed: {record['total_elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
