"""HumanEval+ loader (paper §4.2's secondary contamination-suspect condition,
164-item hard ceiling).

Thin wrapper around `evalplus.data.get_human_eval_plus`. The full evalplus
problem dict (canonical_solution, base_input, plus_input, atol, entry_point,
contract) is kept in `Item.metadata["evalplus_problem"]` rather than
flattened into the shared schema, since `scoring/sandbox.py` is the only
consumer and needs the dict shape evalplus's own `untrusted_check`/
`trusted_exec` expect verbatim.
"""

from __future__ import annotations

from qcd.constants import HUMANEVAL_N_ITEMS
from qcd.data.schema import Dataset, Item

EVALPLUS_DATASET_NAME = "humaneval"


def load_humaneval() -> list[Item]:
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash  # noqa: PLC0415

    problems = get_human_eval_plus()
    release_version = get_human_eval_plus_hash()

    if len(problems) != HUMANEVAL_N_ITEMS:
        raise RuntimeError(
            f"HumanEval+ returned {len(problems)} items, expected exactly "
            f"{HUMANEVAL_N_ITEMS} (constants.HUMANEVAL_N_ITEMS) — the installed "
            "evalplus version's item count has drifted from the paper's hard "
            "ceiling; do not silently proceed (pipeline_build_plan.md open "
            "assumption #3)."
        )

    items = []
    for task_id, problem in problems.items():
        items.append(
            Item(
                item_id=task_id,
                dataset=Dataset.HUMANEVAL,
                prompt=problem["prompt"],
                release_version=release_version,
                metadata={"evalplus_problem": problem, "evalplus_dataset_name": EVALPLUS_DATASET_NAME},
            )
        )
    return items
