from qcd.data.schema import Dataset, Item
from qcd.ground_truth.string_match import MatchConfig, extract_text, normalize_text, scan_corpus


def _item(item_id: str, prompt: str) -> Item:
    return Item(item_id=item_id, dataset=Dataset.HUMANEVAL, prompt=prompt)


def test_normalized_verbatim_tolerates_case_unicode_and_whitespace():
    item = _item("exact", "Write Ａ function\nthat adds two numbers.")
    rows = scan_corpus(
        [item], [{"id": "doc-1", "text": "PREFIX write a FUNCTION that adds two numbers. suffix"}],
        corpus_name="synthetic", stage="sft", config=MatchConfig(3, 0.8),
    )
    assert normalize_text("Ａ") == "a"
    assert rows[0]["normalized_verbatim"] is True
    assert rows[0]["string_match_label"] is True
    assert rows[0]["document_id"] == "doc-1"


def test_ngram_overlap_can_retrieve_near_verbatim_without_exact_match():
    item = _item("near", "alpha beta gamma delta epsilon zeta")
    [row] = scan_corpus(
        [item], [{"id": "doc-2", "text": "alpha beta gamma delta epsilon CHANGED"}],
        corpus_name="synthetic", stage="pretraining", config=MatchConfig(2, 0.8),
    )
    assert row["normalized_verbatim"] is False
    assert row["ngram_coverage"] == 0.8
    assert row["string_match_label"] is True


def test_absent_item_emits_auditable_negative_row():
    [row] = scan_corpus(
        [_item("none", "one two three four")], [{"id": "doc", "text": "unrelated corpus text"}],
        corpus_name="synthetic", stage="rlvr", config=MatchConfig(2, 0.5),
    )
    assert row["string_match_label"] is False
    assert row["document_id"] is None
    assert row["documents_scanned"] == 1


def test_extract_text_flattens_sft_dpo_and_rl_nested_values():
    row = {
        "messages": [{"role": "user", "content": "prompt"}],
        "chosen": [{"role": "assistant", "content": "answer"}],
        "ground_truth": ["truth"],
        "number": 3,
    }
    assert extract_text(row).splitlines() == ["prompt", "answer", "truth"]


def test_same_item_id_in_different_datasets_does_not_collide():
    items = [
        Item(item_id="shared", dataset=Dataset.HUMANEVAL, prompt="alpha beta gamma delta"),
        Item(item_id="shared", dataset=Dataset.MBPPPLUS, prompt="one two three four"),
    ]
    rows = scan_corpus(
        items,
        [{"id": "doc", "text": "alpha beta gamma delta"}],
        corpus_name="synthetic",
        stage="sft",
        config=MatchConfig(2, 0.8),
    )

    assert [(row["dataset"], row["string_match_label"]) for row in rows] == [
        ("humaneval", True),
        ("mbppplus", False),
    ]


def test_progress_callback_reports_completed_intervals():
    progress = []
    scan_corpus(
        [_item("x", "alpha beta gamma")],
        ({"text": "unrelated"} for _ in range(5)),
        corpus_name="synthetic",
        stage="sft",
        config=MatchConfig(2, 0.8),
        progress_every=2,
        progress_callback=progress.append,
    )
    assert progress == [2, 4]
