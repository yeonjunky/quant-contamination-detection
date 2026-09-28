#!/usr/bin/env python
"""Deprecated compatibility entry point.

Engineering-validation rows are not a scientific pilot and must not be turned
into effect-size, power, or CDD-gate outputs. The frozen main-study analysis
will use a separately validated analysis command after ``run_main.py``.
"""


def main() -> None:
    raise SystemExit(
        "aggregate_pilot.py has been retired: validation data must not be "
        "aggregated as manuscript evidence or used to change the study design."
    )


if __name__ == "__main__":
    main()
