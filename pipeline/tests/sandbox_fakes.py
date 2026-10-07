"""Stand-ins for `real_run._timed_partial_pass_rate`.

They run in the driver's spawned sandbox workers, so they live in an
importable module (a worker unpickles a function by its module and name) and
report back through the file named by `$QCD_TEST_SANDBOX_LOG`, since a worker
shares no memory with the test.
"""

import os
import sys
import time

# The scoring time each fake reports, so a test can tell the worker's own
# figure from one measured around the main process's wait.
SECONDS = 0.125


def _log(*fields) -> None:
    with open(os.environ["QCD_TEST_SANDBOX_LOG"], "a", encoding="utf-8") as log:
        log.write(" ".join(str(f) for f in fields) + "\n")


def _rate(item) -> float:
    return int(item.item_id[1:]) / 10


def passes(item, candidate_code):
    return 1.0, SECONDS


def records(item, candidate_code):
    # A forked worker inherits the test module; a spawned one never imports it.
    _log(item.item_id, os.getpid(), "tests.test_real_run" in sys.modules)
    return _rate(item), SECONDS


def slower_for_earlier_items(item, candidate_code):
    time.sleep(0.15 * (5 - int(item.item_id[1:])))
    _log(item.item_id, time.time())
    return _rate(item), SECONDS


def fails_on_q2(item, candidate_code):
    if item.item_id == "q2":
        raise ValueError("sandbox failed on q2")
    return _rate(item), SECONDS
