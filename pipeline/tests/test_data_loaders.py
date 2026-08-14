"""Real (network-hitting, GPU-free) tests for the dataset loaders. No mock
here — these assert the load-bearing facts found during pipeline
construction (pipeline_build_plan.md open assumptions #3/#4): evalplus's
exact item counts, and that LiveCodeBench's raw jsonl files actually split
correctly by contest_date.
"""

import datetime as dt

from qcd.constants import HUMANEVAL_N_ITEMS, MBPPPLUS_N_ITEMS
from qcd.data.humaneval import load_humaneval
from qcd.data.livecodebench import load_livecodebench_split
from qcd.data.mbppplus import load_mbppplus
from qcd.data.schema import Dataset


def test_humaneval_exact_count():
    items = load_humaneval()
    assert len(items) == HUMANEVAL_N_ITEMS == 164
    assert all(item.dataset is Dataset.HUMANEVAL for item in items)
    assert all(item.contamination_proxy is True for item in items)


def test_humaneval_items_carry_evalplus_problem():
    items = load_humaneval()
    problem = items[0].metadata["evalplus_problem"]
    assert {"canonical_solution", "base_input", "plus_input", "atol", "entry_point"} <= problem.keys()


def test_mbppplus_exact_count():
    items = load_mbppplus()
    assert len(items) == MBPPPLUS_N_ITEMS == 378
    assert all(item.dataset is Dataset.MBPPPLUS for item in items)
    assert all(item.contamination_proxy is True for item in items)


def test_livecodebench_split_nonempty_and_dated_correctly():
    # Mid-range boundary date: known to fall inside test.jsonl's date span
    # (2023-05-07..2024-03-02), so both sides of the split are guaranteed
    # non-trivial without downloading every release file.
    boundary = dt.datetime(2023, 12, 1)
    pre, post = load_livecodebench_split(boundary, release_version="release_v1")

    assert len(pre) > 0
    assert len(post) > 0
    assert all(item.dataset is Dataset.LCB_PRE for item in pre)
    assert all(item.dataset is Dataset.LCB_POST for item in post)
    assert all(item.contamination_proxy is True for item in pre)
    assert all(item.contamination_proxy is False for item in post)

    for item in pre:
        assert dt.datetime.fromisoformat(item.metadata["contest_date"]) < boundary
    for item in post:
        assert dt.datetime.fromisoformat(item.metadata["contest_date"]) >= boundary


def test_livecodebench_unknown_release_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown LiveCodeBench release"):
        load_livecodebench_split(dt.datetime(2024, 1, 1), release_version="release_v99")
