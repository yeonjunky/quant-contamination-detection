"""Dispatches to the right sandbox harness (scoring/sandbox.py) by dataset
family and returns a single fractional partial-pass-rate — the shared
interface the rest of the pipeline (io/raw_writer.py and validation diagnostics)
scores against, so callers never need to know which harness backs a given
item's `Dataset`.
"""

from __future__ import annotations

from qcd.data.schema import Dataset, Item
from qcd.scoring import sandbox


def partial_pass_rate(item: Item, candidate_code: str) -> float:
    if item.dataset in (Dataset.HUMANEVAL, Dataset.MBPPPLUS):
        return sandbox.evalplus_partial_pass(item, candidate_code)
    if item.dataset in (Dataset.LCB_PRE, Dataset.LCB_POST):
        return sandbox.livecodebench_partial_pass(item, candidate_code)
    raise ValueError(f"no scoring harness registered for dataset {item.dataset!r}")
