#!/usr/bin/env python
"""Deprecated compatibility entry point.

The former ``run_pilot.py`` mixed engineering validation with a scientific
pilot: it ran a small real-model sample and immediately converted those rows
into effect-size, power, and CDD-gate outputs. That workflow is no longer part
of the study design. Validation observations must not determine analysis
eligibility, sample size, or manuscript results.

Use ``run_dry_run.py`` for GPU-free implementation validation,
``run_smoke_test.py``/``run_lcb_smoke_test.py`` for bounded H100 wiring checks,
and ``run_main.py`` for the frozen study execution.
"""


def main() -> None:
    raise SystemExit(
        "run_pilot.py has been retired: validation-only runs must not produce "
        "scientific pilot statistics. Use run_dry_run.py, run_smoke_test.py, "
        "or run_main.py as appropriate."
    )


if __name__ == "__main__":
    main()
