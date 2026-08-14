"""Real, non-mock, known-answer tests for scoring/sandbox.py — no GPU or
model needed, just a subprocess and (for HumanEval+/MBPP+) evalplus's own
ground-truth machinery. Canonical solutions must score close to 1.0;
deliberately broken solutions must score low. This is the first place in
the suite that actually executes generated-looking code, which is exactly
what the real pipeline will do at scale.
"""

import json

from qcd.data.humaneval import load_humaneval
from qcd.data.schema import Dataset, Item
from qcd.scoring.pass_rate import partial_pass_rate
from qcd.scoring.sandbox import evalplus_partial_pass, livecodebench_partial_pass


def _humaneval_item(task_id: str) -> Item:
    items = {item.item_id: item for item in load_humaneval()}
    return items[task_id]


def test_evalplus_canonical_solution_passes_fully():
    item = _humaneval_item("HumanEval/0")
    problem = item.metadata["evalplus_problem"]
    candidate = problem["prompt"] + problem["canonical_solution"]

    rate = evalplus_partial_pass(item, candidate)

    assert rate == 1.0


def test_evalplus_broken_solution_scores_low():
    item = _humaneval_item("HumanEval/0")
    problem = item.metadata["evalplus_problem"]
    entry_point = problem["entry_point"]
    broken = problem["prompt"] + f"    return False\n"
    assert entry_point == "has_close_elements"

    rate = evalplus_partial_pass(item, broken)

    assert 0.0 <= rate < 1.0


def test_partial_pass_rate_dispatches_to_evalplus_for_humaneval():
    item = _humaneval_item("HumanEval/0")
    problem = item.metadata["evalplus_problem"]
    candidate = problem["prompt"] + problem["canonical_solution"]

    assert partial_pass_rate(item, candidate) == 1.0


# --- LiveCodeBench stdin harness --------------------------------------------


def _stdin_item(public_cases: list[dict]) -> Item:
    return Item(
        item_id="synthetic-stdin",
        dataset=Dataset.LCB_PRE,
        prompt="Double the input integer.",
        metadata={"public_test_cases": json.dumps(public_cases), "private_test_cases": None},
    )


def test_livecodebench_stdin_harness_all_pass():
    item = _stdin_item(
        [
            {"input": "5\n", "output": "10\n", "testtype": "stdin"},
            {"input": "3\n", "output": "6\n", "testtype": "stdin"},
        ]
    )
    program = "n = int(input())\nprint(n * 2)\n"

    rate = livecodebench_partial_pass(item, program)

    assert rate == 1.0


def test_livecodebench_stdin_harness_partial_credit():
    item = _stdin_item(
        [
            {"input": "5\n", "output": "10\n", "testtype": "stdin"},
            {"input": "3\n", "output": "6\n", "testtype": "stdin"},
        ]
    )
    # Off-by-one bug: only correct for n=5, wrong for n=3.
    program = "n = int(input())\nprint(n * 2 if n == 5 else n * 3)\n"

    rate = livecodebench_partial_pass(item, program)

    assert rate == 0.5


def test_livecodebench_stdin_harness_timeout_counts_as_fail():
    item = _stdin_item([{"input": "5\n", "output": "10\n", "testtype": "stdin"}])
    program = "import time\ntime.sleep(5)\n"

    rate = livecodebench_partial_pass(item, program, timeout=0.5)

    assert rate == 0.0


# --- LiveCodeBench functional harness ---------------------------------------


def _functional_item(public_cases: list[dict], func_name: str) -> Item:
    return Item(
        item_id="synthetic-functional",
        dataset=Dataset.LCB_POST,
        prompt="Add two numbers.",
        metadata={
            "public_test_cases": json.dumps(public_cases),
            "private_test_cases": None,
            "func_name": func_name,
        },
    )


def test_livecodebench_functional_harness_all_pass():
    item = _functional_item(
        [
            {"input": "2\n3\n", "output": "5", "testtype": "functional"},
            {"input": "10\n-4\n", "output": "6", "testtype": "functional"},
        ],
        func_name="add",
    )
    solution = "class Solution:\n    def add(self, a, b):\n        return a + b\n"

    rate = livecodebench_partial_pass(item, solution)

    assert rate == 1.0


def test_livecodebench_functional_harness_partial_credit():
    item = _functional_item(
        [
            {"input": "2\n3\n", "output": "5", "testtype": "functional"},
            {"input": "10\n-4\n", "output": "6", "testtype": "functional"},
        ],
        func_name="add",
    )
    # Wrong for negative operands.
    solution = "class Solution:\n    def add(self, a, b):\n        return abs(a) + abs(b)\n"

    rate = livecodebench_partial_pass(item, solution)

    assert rate == 0.5
