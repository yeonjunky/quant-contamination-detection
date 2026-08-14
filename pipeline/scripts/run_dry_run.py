#!/usr/bin/env python
"""Local mock spike, interactive. Thin CLI shim over qcd.dry_run.main() —
the actual pipeline logic lives in the installed package (qcd/dry_run.py)
so tests/test_mock_pipeline_end_to_end.py can import and call
qcd.dry_run.run_dry_run() directly, without sys.path surgery to reach into
a bare scripts/ directory.

Usage: python scripts/run_dry_run.py
"""

from qcd.dry_run import main

if __name__ == "__main__":
    main()
