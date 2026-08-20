#!/usr/bin/env python
"""Rebuild pilot_summary.json and power_recompute.json from saved parquet."""

import argparse
from pathlib import Path

from qcd.pilot.aggregate import aggregate_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=Path("data/raw/pilot"))
    parser.add_argument("--baseline", default="bf16")
    parser.add_argument("--target", default="bnb_nf4")
    args = parser.parse_args()
    summary, _ = aggregate_pilot(args.run_dir, baseline=args.baseline, target=args.target)
    print(f"Wrote {args.run_dir / 'pilot_summary.json'}")
    print(f"Wrote {args.run_dir / 'power_recompute.json'}")
    print(f"CDD gates: {summary['cdd_gate']}")


if __name__ == "__main__":
    main()
