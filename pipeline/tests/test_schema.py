import dataclasses

import pytest

from qcd.data.schema import (
    CorpusEvidenceFamily, CorpusReferenceRecord, CorpusReferenceStatus, Dataset, Item,
    TemporalProxyLabel, corpus_reference_from_json, corpus_reference_to_json,
)


def _item(dataset: Dataset) -> Item:
    return Item(item_id="x", dataset=dataset, prompt="def solve(): ...")


def _item_with(dataset: Dataset, records) -> Item:
    return dataclasses.replace(_item(dataset), corpus_reference=records)


def _record(model, status, *, family=CorpusEvidenceFamily.INSTANCE_STRING, stage="pretraining"):
    return CorpusReferenceRecord(
        model=model, method_family=family.value, status=status,
        corpus="allenai/dolma3_mix-6T-1025-7B", corpus_revision="2ca900fb", stage=stage,
    )


def test_contamination_proxy_matches_paper_condition_table():
    assert _item(Dataset.LCB_PRE).contamination_proxy is True
    assert _item(Dataset.LCB_POST).contamination_proxy is False
    assert _item(Dataset.HUMANEVAL).contamination_proxy is True
    assert _item(Dataset.MBPPPLUS).contamination_proxy is True


def test_corpus_reference_axis_defaults_to_unmeasured():
    """Empty is "not searched", which is not `not-observable` — that value is a
    recorded statement that no corpus can be searched (paper §4.2)."""
    item = _item(Dataset.LCB_PRE)
    assert item.corpus_reference == ()
    assert item.corpus_status("Olmo3-7B-Instruct", CorpusEvidenceFamily.INSTANCE_STRING) is None


def test_corpus_status_is_per_model_not_one_olmo3_value():
    """E-F14: Olmo3-7B and Olmo3.1-32B have different pretraining corpora, so a
    match in one arm's corpus is not a match in the other's."""
    item = _item_with(
        Dataset.LCB_PRE,
        (
            _record("Olmo3-7B-Instruct", CorpusReferenceStatus.CONFIRMED_MATCH),
            _record("Olmo3.1-32B-Instruct", CorpusReferenceStatus.NO_MATCH_FOUND),
            _record("Qwen2.5-7B-Instruct", CorpusReferenceStatus.NOT_OBSERVABLE),
        ),
    )
    family = CorpusEvidenceFamily.INSTANCE_STRING
    assert item.corpus_status("Olmo3-7B-Instruct", family) is CorpusReferenceStatus.CONFIRMED_MATCH
    assert item.corpus_status("Olmo3.1-32B-Instruct", family) is CorpusReferenceStatus.NO_MATCH_FOUND
    assert item.corpus_status("Qwen2.5-7B-Instruct", family) is CorpusReferenceStatus.NOT_OBSERVABLE
    assert item.corpus_status("Llama-3.1-8B-Instruct", family) is None


def test_corpus_status_is_reported_per_method_family():
    """Paper §5 step 5 reports the three counts "from each family separately"."""
    item = _item_with(
        Dataset.HUMANEVAL,
        (
            _record("Olmo3-7B-Instruct", CorpusReferenceStatus.NO_MATCH_FOUND),
            _record(
                "Olmo3-7B-Instruct", CorpusReferenceStatus.CONFIRMED_MATCH,
                family=CorpusEvidenceFamily.PARAPHRASE,
            ),
        ),
    )
    assert item.corpus_status(
        "Olmo3-7B-Instruct", CorpusEvidenceFamily.INSTANCE_STRING
    ) is CorpusReferenceStatus.NO_MATCH_FOUND
    assert item.corpus_status(
        "Olmo3-7B-Instruct", CorpusEvidenceFamily.PARAPHRASE
    ) is CorpusReferenceStatus.CONFIRMED_MATCH
    assert item.corpus_status("Olmo3-7B-Instruct", CorpusEvidenceFamily.SURFACE_PROGRAM) is None


def test_partially_searched_stages_do_not_become_no_match_found():
    """A completed stage plus an unsearched one is `not-observable`, never
    `no-match-found` — §4.2: "`no-match-found` is not renamed `clean`"."""
    item = _item_with(
        Dataset.LCB_POST,
        (
            _record("Olmo3-7B-Instruct", CorpusReferenceStatus.NO_MATCH_FOUND, stage="sft"),
            _record("Olmo3-7B-Instruct", CorpusReferenceStatus.NOT_OBSERVABLE, stage="pretraining"),
        ),
    )
    assert item.corpus_status(
        "Olmo3-7B-Instruct", CorpusEvidenceFamily.INSTANCE_STRING
    ) is CorpusReferenceStatus.NOT_OBSERVABLE


def test_corpus_reference_record_rejects_an_unknown_method_family():
    with pytest.raises(ValueError, match="method_family"):
        CorpusReferenceRecord(
            model="Olmo3-7B-Instruct", method_family="tracer",
            status=CorpusReferenceStatus.CONFIRMED_MATCH,
        )


def test_corpus_reference_round_trips_through_the_parquet_column():
    records = (
        _record("Olmo3-7B-Instruct", CorpusReferenceStatus.CONFIRMED_MATCH),
        _record("Olmo3.1-32B-Instruct", CorpusReferenceStatus.NOT_OBSERVABLE),
    )
    assert corpus_reference_from_json(corpus_reference_to_json(records)) == records


def test_a_raw_tree_written_before_this_column_existed_still_reads():
    """Old `items.parquet` files carry a `tracer_label` float and no
    `corpus_reference_json`; pandas hands back None or NaN for the missing
    value and both must read as "not measured"."""
    assert corpus_reference_from_json(None) == ()
    assert corpus_reference_from_json(float("nan")) == ()
    assert corpus_reference_from_json("") == ()
    assert corpus_reference_from_json("[]") == ()


def test_dataset_enum_values_are_stable_strings():
    # These strings are the on-disk/parquet representation; changing them is
    # a data-format break, not just a rename.
    assert Dataset.LCB_PRE.value == "lcb_pre"
    assert Dataset.LCB_POST.value == "lcb_post"
    assert Dataset.HUMANEVAL.value == "humaneval"
    assert Dataset.MBPPPLUS.value == "mbppplus"


def test_proxy_and_corpus_status_values_are_explicit_non_binary_strings():
    assert TemporalProxyLabel.POSSIBLE_EXPOSURE.value == "possible-exposure"
    assert TemporalProxyLabel.SHARED_CLEAN_CONTROL.value == "shared-clean-control"
    assert CorpusReferenceStatus.CONFIRMED_MATCH.value == "confirmed-match"
    assert CorpusReferenceStatus.NO_MATCH_FOUND.value == "no-match-found"
    assert CorpusReferenceStatus.NOT_OBSERVABLE.value == "not-observable"
