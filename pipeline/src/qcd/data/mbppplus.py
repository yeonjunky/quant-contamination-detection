"""MBPP+ loader (paper §4.2's secondary contamination-suspect condition, a
separate arm — never pooled with HumanEval; see analysis/aggregation.py's
pooling guard).

Same wrapping strategy as data/humaneval.py: thin wrapper around
`evalplus.data.get_mbpp_plus`, full evalplus problem dict kept verbatim in
`Item.metadata["evalplus_problem"]` for scoring/sandbox.py.
"""

from __future__ import annotations

from qcd.constants import MBPPPLUS_N_ITEMS
from qcd.data.schema import Dataset, Item

EVALPLUS_DATASET_NAME = "mbpp"


def load_mbppplus() -> list[Item]:
    from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash  # noqa: PLC0415

    problems = get_mbpp_plus()
    release_version = get_mbpp_plus_hash()

    if len(problems) != MBPPPLUS_N_ITEMS:
        raise RuntimeError(
            f"MBPP+ returned {len(problems)} items, expected exactly "
            f"{MBPPPLUS_N_ITEMS} (constants.MBPPPLUS_N_ITEMS) — the installed "
            "evalplus version's item count has drifted from the paper's "
            "expected size; do not silently proceed (pipeline_build_plan.md "
            "open assumption #3)."
        )

    items = []
    for task_id, problem in problems.items():
        items.append(
            Item(
                item_id=task_id,
                dataset=Dataset.MBPPPLUS,
                prompt=problem["prompt"],
                release_version=release_version,
                metadata={"evalplus_problem": problem, "evalplus_dataset_name": EVALPLUS_DATASET_NAME},
            )
        )
    return items
