import pytest

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
    assert rows[0]["match_detected"] is True
    assert rows[0]["corpus_status"] == "confirmed-match"
    assert rows[0]["document_id"] == "doc-1"


def test_ngram_overlap_can_retrieve_near_verbatim_without_exact_match():
    item = _item("near", "alpha beta gamma delta epsilon zeta")
    [row] = scan_corpus(
        [item], [{"id": "doc-2", "text": "alpha beta gamma delta epsilon CHANGED"}],
        corpus_name="synthetic", stage="pretraining", config=MatchConfig(2, 0.8),
    )
    assert row["normalized_verbatim"] is False
    assert row["ngram_coverage"] == 0.8
    assert row["match_detected"] is True
    assert row["corpus_status"] == "confirmed-match"


def test_absent_item_emits_no_match_status_only_after_complete_scan():
    [row] = scan_corpus(
        [_item("none", "one two three four")], [{"id": "doc", "text": "unrelated corpus text"}],
        corpus_name="synthetic", stage="rlvr", config=MatchConfig(2, 0.5),
        coverage_complete=True,
    )
    assert row["match_detected"] is False
    assert row["corpus_status"] == "no-match-found"
    assert row["coverage_complete"] is True
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
        coverage_complete=True,
    )

    assert [(row["dataset"], row["corpus_status"]) for row in rows] == [
        ("humaneval", "confirmed-match"),
        ("mbppplus", "no-match-found"),
    ]


def test_absent_item_in_incomplete_scan_is_not_observable():
    [row] = scan_corpus(
        [_item("none", "one two three four")],
        [{"id": "doc", "text": "unrelated corpus text"}],
        corpus_name="synthetic", stage="rlvr", config=MatchConfig(2, 0.5),
        coverage_complete=False,
    )
    assert row["match_detected"] is False
    assert row["corpus_status"] == "not-observable"


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


def test_candidate_mode_keeps_ranked_top_k_with_retrieval_text_and_metadata():
    item = _item("ranked", "alpha beta gamma delta epsilon")
    rows = scan_corpus(
        [item],
        [
            {"id": "weak", "text": "alpha beta unrelated", "source": "web", "version": "v1"},
            {
                "id": "strong", "text": "prefix alpha beta gamma delta changed suffix",
                "source": "code", "version": "v2", "created": "2024-01-01",
            },
            {"id": "middle", "text": "alpha beta gamma changed"},
        ],
        corpus_name="synthetic", stage="pretraining", config=MatchConfig(2, 0.8),
        top_k=2, evidence_only=True, include_document_text=True,
    )

    assert [row["document_id"] for row in rows] == ["strong", "middle"]
    assert [row["candidate_rank"] for row in rows] == [1, 2]
    assert rows[0]["document_text"] == "prefix alpha beta gamma delta changed suffix"
    assert rows[0]["normalized_document_text"] == rows[0]["document_text"]
    assert len(rows[0]["document_sha256"]) == 64
    assert rows[0]["document_source"] == "code"
    assert rows[0]["document_version"] == "v2"
    assert rows[0]["matched_token_start"] is not None
    assert rows[0]["match_context"]
    assert rows[0]["document_text_truncated"] is False


def test_candidate_mode_omits_negatives_but_reports_completion_count():
    completed = []
    rows = scan_corpus(
        [_item("none", "alpha beta gamma")],
        [{"text": "unrelated"}, {"text": "also unrelated"}],
        corpus_name="synthetic", stage="pretraining", config=MatchConfig(2, 0.8),
        evidence_only=True, completion_callback=completed.append,
    )
    assert rows == []
    assert completed == [2]


def test_top_k_must_be_positive():
    with pytest.raises(ValueError, match="top_k"):
        scan_corpus(
            [], [], corpus_name="synthetic", stage="pretraining", top_k=0,
        )
