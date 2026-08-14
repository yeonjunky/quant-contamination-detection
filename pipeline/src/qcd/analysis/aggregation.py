"""Pooling guard — paper §4.2: "MBPP+ ... Not pooled with HumanEval —
different difficulty distributions would reintroduce the base-rate confound
*inside* a nominally single condition." CLAUDE.md §5 point 5 states this as
a hard design invariant, not a style preference: "HumanEval과 MBPP+를 풀링하지
말 것."

Every place in the analysis code that groups items by condition must route
through `assert_not_pooled` (or `combined_items`, which calls it) so a
future refactor can't silently reintroduce the base-rate confound §4.5.3
warns about.
"""

from __future__ import annotations

from qcd.data.schema import Dataset, Item

_FORBIDDEN_POOL = frozenset({Dataset.HUMANEVAL, Dataset.MBPPPLUS})


class PooledSecondaryConditionsError(Exception):
    """Raised when HumanEval and MBPP+ items are combined into one analysis
    cell — forbidden by paper §4.2 (n=542 is "a sample-size reference only,
    never a pooled analysis cell")."""


def assert_not_pooled(datasets: set[Dataset] | frozenset[Dataset]) -> None:
    if _FORBIDDEN_POOL <= set(datasets):
        raise PooledSecondaryConditionsError(
            "HumanEval and MBPP+ items were combined into one analysis cell — "
            "forbidden by paper §4.2. Their different difficulty distributions "
            "reintroduce the base-rate confound §4.5.3 quantifies; keep them as "
            "separate arms. n=542 (HumanEval+MBPP+) is a sample-size reference "
            "only, never a pooled analysis cell."
        )


def combined_items(items: list[Item]) -> list[Item]:
    """A guarded convenience for callers that legitimately need items from
    multiple *allowed* conditions in one list (e.g. all four conditions for
    an overall dataset count) — raises before returning anything if the
    forbidden HumanEval+MBPP+ combination is present."""
    assert_not_pooled({item.dataset for item in items})
    return items
