#!/usr/bin/env python
"""Check the main run's code-scoring worker pool on this machine.

Scores the reference solutions of the first N HumanEval+ and MBPP+ items
twice: through the same spawned worker pool the main run uses
(`real_run.start_sandbox_pool`), with all items in flight at once, and one
by one in this process. A reference solution passes every test, so each
item must score 1.0 on both paths. Exits 1 when any item does not.

This is engineering validation of the scoring harness: it scores the
benchmarks' own reference solutions, never model output, and writes no data.

Usage:
  python scripts/check_sandbox_pool.py [--n-items 20]
"""

from __future__ import annotations

import argparse
import sys
import time

from qcd.data.humaneval import load_humaneval
from qcd.data.mbppplus import load_mbppplus
from qcd.real_run import _assemble_candidate_code, _timed_partial_pass_rate, start_sandbox_pool


def reference_code(item) -> str:
    return _assemble_candidate_code(item, item.metadata["evalplus_problem"]["canonical_solution"])


def compare(items, pooled_score, serial_score) -> list[str]:
    """Return one line per item whose pooled or serial score is not 1.0."""
    pooled = pooled_score(items)
    serial = [serial_score(item) for item in items]
    return [
        f"{item.item_id}: pooled {p}, serial {s}"
        for item, p, s in zip(items, pooled, serial)
        if p != 1.0 or s != 1.0
    ]


def _pooled(items) -> list[float]:
    sandbox = start_sandbox_pool()
    try:
        futures = [sandbox.submit(_timed_partial_pass_rate, item, reference_code(item)) for item in items]
        return [future.result()[0] for future in futures]
    finally:
        sandbox.shutdown(cancel_futures=True)


def _serial(item) -> float:
    return _timed_partial_pass_rate(item, reference_code(item))[0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-items", type=int, default=20, help="items per benchmark (default: %(default)s)")
    args = parser.parse_args(argv)

    items = load_humaneval()[: args.n_items] + load_mbppplus()[: args.n_items]
    started = time.perf_counter()
    failures = compare(items, _pooled, _serial)
    print(f"{len(items)} reference solutions scored pooled and serially in {time.perf_counter() - started:.0f} s")
    for line in failures:
        print(f"FAIL  {line}")
    print("PASS" if not failures else f"FAIL  {len(failures)} item(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
