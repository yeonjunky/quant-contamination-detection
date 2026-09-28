import datetime as dt
from collections import Counter

import pytest

from qcd.data.livecodebench import load_livecodebench_split
from qcd.data.schema import Dataset, Item
from qcd.data.temporal_labels import materialize_model_item_labels
from qcd.models.registry import LLAMA3_1_8B, OLMO3_7B, QWEN2_5_7B


def _lcb(item_id: str, contest_date: str) -> Item:
    return Item(
        item_id=item_id,
        dataset=Dataset.LCB_PRE if contest_date < "2025-01-01" else Dataset.LCB_POST,
        prompt="solve",
        metadata={"contest_date": contest_date},
    )


def test_labels_are_materialized_per_model_item_and_shared_control_is_common():
    items = [
        _lcb("ambiguous", "2023-06-01"),
        _lcb("middle", "2024-10-01"),
        _lcb("shared", "2025-02-01"),
        Item("he", Dataset.HUMANEVAL, "def f(): ..."),
    ]
    rows = materialize_model_item_labels(
        items, (QWEN2_5_7B, OLMO3_7B, LLAMA3_1_8B),
        shared_control_boundary=dt.datetime(2025, 1, 1),
    )
    by_key = {(row["model"], row["item_id"]): row for row in rows}

    assert by_key[(QWEN2_5_7B.name, "middle")]["primary_label"] == "clean-by-model-cutoff"
    assert by_key[(QWEN2_5_7B.name, "middle")]["publication_date"] == "2024-10-01"
    assert by_key[(QWEN2_5_7B.name, "middle")]["primary_first_post_date"] == "2024-09-20"
    assert by_key[(QWEN2_5_7B.name, "middle")]["shared_control_start_date"] == "2025-01-01"
    assert by_key[(OLMO3_7B.name, "middle")]["primary_label"] == "possible-exposure"
    assert {
        by_key[(model.name, "shared")]["primary_label"]
        for model in (QWEN2_5_7B, OLMO3_7B, LLAMA3_1_8B)
    } == {"shared-clean-control"}
    assert {
        by_key[(model.name, "he")]["primary_label"]
        for model in (QWEN2_5_7B, OLMO3_7B, LLAMA3_1_8B)
    } == {"possible-exposure"}


def test_llama_sensitivity_boundary_has_no_possible_lcb_exposure_after_collection_start():
    items = [
        _lcb("first", "2023-05-07"),
        _lcb("late", "2024-12-31"),
        _lcb("shared", "2025-01-01"),
    ]
    rows = materialize_model_item_labels(
        items, (LLAMA3_1_8B,), shared_control_boundary=dt.datetime(2025, 1, 1)
    )
    lcb_rows = [row for row in rows if row["dataset"].startswith("lcb_")]
    assert not any(row["sensitivity_label"] == "possible-exposure" for row in lcb_rows)
    assert next(row for row in rows if row["item_id"] == "first")["boundary_ambiguous"] is True


def test_shared_control_must_follow_each_models_primary_boundary():
    with pytest.raises(ValueError, match="precedes"):
        materialize_model_item_labels(
            [_lcb("too-early", "2024-01-01")],
            (QWEN2_5_7B,),
            shared_control_boundary=dt.datetime(2024, 1, 1),
        )


def test_release_v6_primary_model_item_counts_match_frozen_design():
    pre, post = load_livecodebench_split(
        dt.datetime(2025, 1, 1), release_version="release_v6"
    )
    models = (QWEN2_5_7B, OLMO3_7B, LLAMA3_1_8B)
    rows = materialize_model_item_labels(
        [*pre, *post], models, shared_control_boundary=dt.datetime(2025, 1, 1)
    )
    counts = {
        model.name: Counter(
            row["primary_label"] for row in rows if row["model"] == model.name
        )
        for model in models
    }
    assert counts[QWEN2_5_7B.name] == {
        "possible-exposure": 690,
        "clean-by-model-cutoff": 183,
        "shared-clean-control": 182,
    }
    assert counts[LLAMA3_1_8B.name] == {
        "possible-exposure": 326,
        "clean-by-model-cutoff": 547,
        "shared-clean-control": 182,
    }
    assert counts[OLMO3_7B.name] == {
        "possible-exposure": 873,
        "shared-clean-control": 182,
    }
