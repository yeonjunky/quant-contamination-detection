from qcd.data.schema import Dataset, Item


def _item(dataset: Dataset) -> Item:
    return Item(item_id="x", dataset=dataset, prompt="def solve(): ...")


def test_contamination_proxy_matches_paper_condition_table():
    assert _item(Dataset.LCB_PRE).contamination_proxy is True
    assert _item(Dataset.LCB_POST).contamination_proxy is False
    assert _item(Dataset.HUMANEVAL).contamination_proxy is True
    assert _item(Dataset.MBPPPLUS).contamination_proxy is True


def test_tracer_label_defaults_to_unmeasured():
    assert _item(Dataset.LCB_PRE).tracer_label is None


def test_dataset_enum_values_are_stable_strings():
    # These strings are the on-disk/parquet representation; changing them is
    # a data-format break, not just a rename.
    assert Dataset.LCB_PRE.value == "lcb_pre"
    assert Dataset.LCB_POST.value == "lcb_post"
    assert Dataset.HUMANEVAL.value == "humaneval"
    assert Dataset.MBPPPLUS.value == "mbppplus"
