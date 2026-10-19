import pytest

from qcd.data.schema import Dataset, Item
from qcd.ground_truth.string_match import MatchConfig, extract_text, normalize_text, scan_corpus

# Every evidence row is attributed to one checkpoint (paper §4.2's per-model
# corpus axis); the two Olmo arms do not share a pretraining mix.
_MODEL = "Olmo3-7B-Instruct"


def _item(item_id: str, prompt: str) -> Item:
    return Item(item_id=item_id, dataset=Dataset.HUMANEVAL, prompt=prompt)


def test_normalized_verbatim_tolerates_case_unicode_and_whitespace():
    item = _item("exact", "Write Ａ function\nthat adds two numbers.")
    rows = scan_corpus(
        [item], [{"id": "doc-1", "text": "PREFIX write a FUNCTION that adds two numbers. suffix"}],
        model=_MODEL, corpus_name="synthetic", stage="sft", config=MatchConfig(3, 0.8),
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
        model=_MODEL, corpus_name="synthetic", stage="pretraining", config=MatchConfig(2, 0.8),
    )
    assert row["normalized_verbatim"] is False
    assert row["ngram_coverage"] == 0.8
    assert row["match_detected"] is True
    assert row["corpus_status"] == "confirmed-match"


def test_absent_item_emits_no_match_status_only_after_complete_scan():
    [row] = scan_corpus(
        [_item("none", "one two three four")], [{"id": "doc", "text": "unrelated corpus text"}],
        model=_MODEL, corpus_name="synthetic", stage="rlvr", config=MatchConfig(2, 0.5),
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
        model=_MODEL, corpus_name="synthetic",
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
        model=_MODEL, corpus_name="synthetic", stage="rlvr", config=MatchConfig(2, 0.5),
        coverage_complete=False,
    )
    assert row["match_detected"] is False
    assert row["corpus_status"] == "not-observable"


def test_progress_callback_reports_completed_intervals():
    progress = []
    scan_corpus(
        [_item("x", "alpha beta gamma")],
        ({"text": "unrelated"} for _ in range(5)),
        model=_MODEL, corpus_name="synthetic",
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
        model=_MODEL, corpus_name="synthetic", stage="pretraining", config=MatchConfig(2, 0.8),
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
        model=_MODEL, corpus_name="synthetic", stage="pretraining", config=MatchConfig(2, 0.8),
        evidence_only=True, completion_callback=completed.append,
    )
    assert rows == []
    assert completed == [2]


def test_top_k_must_be_positive():
    with pytest.raises(ValueError, match="top_k"):
        scan_corpus(
            [], [], model=_MODEL, corpus_name="synthetic", stage="pretraining", top_k=0,
        )


# --- sensitivity rule (8-gram / 0.5 beside the 13-gram / 0.8 primary) -------


def test_sensitivity_rule_reports_a_separate_status_without_changing_the_primary():
    # Six tokens; the document repeats the first four and changes the rest.
    item = _item("partial", "alpha beta gamma delta epsilon zeta")
    [row] = scan_corpus(
        [item], [{"id": "doc", "text": "alpha beta gamma delta CHANGED CHANGED"}],
        model=_MODEL, corpus_name="synthetic", stage="sft",
        config=MatchConfig(4, 0.8), sensitivity=MatchConfig(2, 0.5),
        coverage_complete=True,
    )
    # Primary: 1 of 3 four-grams. Sensitivity: 3 of 5 bigrams.
    assert row["ngram_coverage"] == pytest.approx(1 / 3)
    assert row["match_detected"] is False
    assert row["corpus_status"] == "no-match-found"
    assert row["sensitivity_ngram_coverage"] == pytest.approx(3 / 5)
    assert row["sensitivity_match_detected"] is True
    assert row["sensitivity_corpus_status"] == "confirmed-match"
    assert (row["sensitivity_ngram_size"], row["sensitivity_ngram_threshold"]) == (2, 0.5)


def test_a_document_sharing_only_a_sensitivity_ngram_is_a_candidate():
    item = _item("short-overlap", "alpha beta gamma delta epsilon zeta")
    [row] = scan_corpus(
        [item], [{"id": "doc", "text": "noise alpha beta noise"}],
        model=_MODEL, corpus_name="synthetic", stage="sft",
        config=MatchConfig(4, 0.8), sensitivity=MatchConfig(2, 0.5),
        coverage_complete=True,
    )
    assert row["document_id"] == "doc"
    assert row["matched_ngrams"] == 0
    assert row["sensitivity_matched_ngrams"] == 1
    assert row["match_context"] is not None


def test_sensitivity_match_outranks_higher_primary_coverage_within_top_k():
    item = _item("rank", "a b c d e f g h")
    documents = [
        # Primary 4-gram coverage 2/5, bigram coverage 4/7: neither rule matches at 0.6.
        {"id": "primary-heavy", "text": "a b c d e x x x"},
        # Primary 0/5, bigram coverage 6/7 by scattered pairs: sensitivity match.
        {"id": "sensitivity-match", "text": "a b x b c x c d x d e x e f x f g"},
    ]
    rows = scan_corpus(
        [item], documents, model=_MODEL, corpus_name="synthetic", stage="sft",
        config=MatchConfig(4, 0.8), sensitivity=MatchConfig(2, 0.6), top_k=1,
    )
    assert [row["document_id"] for row in rows] == ["sensitivity-match"]


def test_without_a_sensitivity_rule_rows_carry_no_sensitivity_fields():
    [row] = scan_corpus(
        [_item("plain", "alpha beta gamma delta")], [{"id": "d", "text": "alpha beta gamma delta"}],
        model=_MODEL, corpus_name="synthetic", stage="sft", config=MatchConfig(2, 0.8),
    )
    assert not any(key.startswith("sensitivity_") for key in row)
