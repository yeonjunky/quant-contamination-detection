import pytest

from qcd.analysis.aggregation import PooledSecondaryConditionsError, assert_not_pooled, combined_items
from qcd.data.schema import Dataset, Item


def _item(dataset: Dataset) -> Item:
    return Item(item_id="x", dataset=dataset, prompt="p")


def test_pooling_humaneval_and_mbppplus_raises():
    with pytest.raises(PooledSecondaryConditionsError):
        assert_not_pooled({Dataset.HUMANEVAL, Dataset.MBPPPLUS})


def test_humaneval_alone_is_fine():
    assert_not_pooled({Dataset.HUMANEVAL})


def test_all_four_conditions_together_is_forbidden_too():
    with pytest.raises(PooledSecondaryConditionsError):
        assert_not_pooled({Dataset.HUMANEVAL, Dataset.MBPPPLUS, Dataset.LCB_PRE, Dataset.LCB_POST})


def test_lcb_pre_and_post_together_is_fine():
    # Only the HumanEval+MBPP+ combination is forbidden — LCB pre/post are
    # meant to be compared against each other.
    assert_not_pooled({Dataset.LCB_PRE, Dataset.LCB_POST})


def test_combined_items_raises_on_forbidden_pool():
    items = [_item(Dataset.HUMANEVAL), _item(Dataset.MBPPPLUS)]
    with pytest.raises(PooledSecondaryConditionsError):
        combined_items(items)


def test_combined_items_passes_through_when_allowed():
    items = [_item(Dataset.LCB_PRE), _item(Dataset.LCB_POST)]
    assert combined_items(items) == items
