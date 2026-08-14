"""Sandboxed execution of model-generated code, behind one function per
dataset family — the fractional "partial test-case pass rate" paper §5 step
1 needs (`passing_count / total_test_count`), not a 0/1 pass@1.

Two execution strategies:

(a) **HumanEval+/MBPP+** — delegates to `evalplus.eval.untrusted_check`
    (`trusted_exec` for the canonical-solution ground truth, matching
    `evalplus.evaluate.get_groundtruth`'s own recipe) rather than
    reimplementing sandboxing: it's the same battle-tested containment
    (`reliability_guard`/`time_limit`/`swallow_io`) evalplus's own CLI uses.
(b) **LiveCodeBench** — a custom subprocess harness, since its test format
    isn't evalplus-shaped. Two sub-harnesses selected by each test case's
    own `testtype` field (confirmed during pipeline construction to be
    uniform per item): `stdin` (Codeforces/AtCoder — pipe input, diff
    stdout) and `functional` (LeetCode — call a `Solution` method with
    parsed args, compare the JSON-serialized return value).

`private_test_cases` is `base64(zlib.compress(pickle.dumps(json_string)))`
— LiveCodeBench's own encoding, confirmed by direct decode against a live
dataset row during pipeline construction, not guessed. `pickle.loads` is a
genuine trust-boundary operation; it's applied here only to LiveCodeBench's
own official HF dataset release, the single source this design already
depends on end-to-end for the LCB pre/post-cutoff split — not to arbitrary
network input.

**Known gaps, flagged rather than silently assumed away:** no network-
namespace isolation on the LCB subprocess path (evalplus's own
`reliability_guard`, used for HumanEval+/MBPP+, disables some but not all
network-capable calls; the LCB path has no equivalent yet); no float-
tolerance comparison for functional-testtype outputs (exact JSON equality
only); no memory-limit enforcement beyond the OS default. Fine for the
paper's own designed use (models generating solutions to fixed benchmark
problems, not adversarial code), but not a hardened untrusted-code sandbox.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import subprocess
import sys
import zlib

from qcd.data.schema import Dataset, Item

# evalplus's reliability_guard() calls resource.setrlimit(RLIMIT_AS, ...) to
# cap subprocess memory (evalplus/eval/utils.py). On macOS this raises
# "ValueError: current limit exceeds maximum limit" — macOS doesn't support
# RLIMIT_AS as an enforceable hard cap the way Linux does, confirmed by
# hitting this during pipeline construction on the actual dev machine.
# EVALPLUS_MAX_MEMORY_BYTES=-1 is evalplus's own documented escape hatch
# (evalplus/eval/__init__.py's query_maximum_memory_bytes: -1 -> None ->
# reliability_guard skips the setrlimit calls entirely). Only set a default
# on macOS, and only if the caller hasn't already set it — the H100/Linux
# box should keep the real memory cap.
if sys.platform == "darwin":
    os.environ.setdefault("EVALPLUS_MAX_MEMORY_BYTES", "-1")

_LCB_HARNESS_TIMEOUT_SECONDS = 6.0

_FUNCTIONAL_HARNESS_PREAMBLE = (
    "from typing import List, Optional, Tuple, Dict, Set, Any\n"
    "import collections, itertools, functools, heapq, bisect, math, re, sys, json\n"
)

_FUNCTIONAL_HARNESS_ENTRYPOINT = """
if __name__ == "__main__":
    _raw = sys.stdin.read()
    _lines = [_l for _l in _raw.split("\\n") if _l != ""]
    _args = [json.loads(_l) for _l in _lines]
    _sol = Solution()
    _result = getattr(_sol, {func_name!r})(*_args)
    print(json.dumps(_result))
"""


# --- HumanEval+ / MBPP+ (evalplus-backed) -----------------------------------


def evalplus_partial_pass(item: Item, candidate_code: str) -> float:
    """`candidate_code` is the full program to exec — for evalplus items,
    that's `item.prompt + completion` (matching
    `evalplus.evaluate.get_groundtruth`'s own
    `problem["prompt"] + problem["canonical_solution"]` convention for the
    reference solution)."""
    from evalplus.eval import untrusted_check  # noqa: PLC0415

    problem = item.metadata["evalplus_problem"]
    dataset_name = item.metadata["evalplus_dataset_name"]
    entry_point = problem["entry_point"]
    atol = problem["atol"]

    expected_base, time_base = _trusted_exec_with_time(problem["prompt"] + problem["canonical_solution"], problem["base_input"], entry_point)
    expected_plus, time_plus = _trusted_exec_with_time(problem["prompt"] + problem["canonical_solution"], problem["plus_input"], entry_point)

    _, details_base = untrusted_check(
        dataset_name, candidate_code, problem["base_input"], entry_point,
        expected=expected_base, atol=atol, ref_time=time_base,
    )
    _, details_plus = untrusted_check(
        dataset_name, candidate_code, problem["plus_input"], entry_point,
        expected=expected_plus, atol=atol, ref_time=time_plus,
    )

    # `details` is truncated to however many inputs were actually attempted
    # before a crash/timeout (see evalplus.eval.untrusted_check), so the
    # denominator must come from the full input lists, not len(details) —
    # an unattempted test counts as failed, not excluded, for a partial
    # pass-rate metric.
    total = len(problem["base_input"]) + len(problem["plus_input"])
    passed = sum(bool(d) for d in details_base) + sum(bool(d) for d in details_plus)
    return passed / total if total else 0.0


def _trusted_exec_with_time(code: str, inputs: list, entry_point: str):
    from evalplus.gen.util import trusted_exec  # noqa: PLC0415

    return trusted_exec(code, inputs, entry_point, record_time=True)


# --- LiveCodeBench (custom subprocess harness) -------------------------------


def _decode_private_test_cases(raw: str) -> list[dict]:
    decompressed = zlib.decompress(base64.b64decode(raw))
    inner_json_string = pickle.loads(decompressed)  # noqa: S301 — see module docstring's trust-boundary note
    return json.loads(inner_json_string)


def _load_test_cases(item: Item) -> list[dict]:
    public = json.loads(item.metadata["public_test_cases"]) if item.metadata.get("public_test_cases") else []
    private_raw = item.metadata.get("private_test_cases")
    private = _decode_private_test_cases(private_raw) if private_raw else []
    return public + private


def livecodebench_partial_pass(item: Item, candidate_code: str, *, timeout: float = _LCB_HARNESS_TIMEOUT_SECONDS) -> float:
    if item.dataset not in (Dataset.LCB_PRE, Dataset.LCB_POST):
        raise ValueError(f"livecodebench_partial_pass called on non-LCB item {item.item_id!r} (dataset={item.dataset})")

    test_cases = _load_test_cases(item)
    if not test_cases:
        raise ValueError(f"item {item.item_id!r} has no test cases (public+private both empty)")

    testtype = test_cases[0]["testtype"]
    if testtype == "functional":
        passed = sum(_run_functional_test(item, candidate_code, tc, timeout) for tc in test_cases)
    elif testtype == "stdin":
        passed = sum(_run_stdin_test(candidate_code, tc, timeout) for tc in test_cases)
    else:
        raise ValueError(f"unknown LiveCodeBench testtype {testtype!r} for item {item.item_id!r}")

    return passed / len(test_cases)


def _run_stdin_test(candidate_code: str, test_case: dict, timeout: float) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-c", candidate_code],
            input=test_case["input"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    if result.returncode != 0:
        return False
    return _normalize_stdout(result.stdout) == _normalize_stdout(test_case["output"])


def _normalize_stdout(text: str) -> list[str]:
    # Strip trailing whitespace per line and drop trailing blank lines —
    # matches the tolerance level of a typical competitive-programming
    # judge, without attempting numeric/whitespace-collapsing tolerance.
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _run_functional_test(item: Item, candidate_code: str, test_case: dict, timeout: float) -> bool:
    func_name = item.metadata.get("func_name")
    if not func_name:
        raise ValueError(f"item {item.item_id!r} is a functional-testtype LCB item with no func_name in metadata")

    script = _FUNCTIONAL_HARNESS_PREAMBLE + candidate_code + "\n" + _FUNCTIONAL_HARNESS_ENTRYPOINT.format(func_name=func_name)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            input=test_case["input"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    if result.returncode != 0:
        return False
    try:
        actual = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return False
    expected = json.loads(test_case["output"])
    return actual == expected
