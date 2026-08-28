import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from qcd.ground_truth.shard_manifest import claim_next, connect, initialize, mark_complete


SCRIPT = Path(__file__).parents[1] / "scripts" / "manage_olmo_pretraining_scan.py"
SPEC = importlib.util.spec_from_file_location("manage_olmo_pretraining_scan", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _candidate(document_id, coverage, *, exact=False, matched=1):
    match_detected = exact or coverage >= 0.8
    return {
        "item_id": "item-1",
        "dataset": "humaneval",
        "document_id": document_id,
        "normalized_verbatim": exact,
        "ngram_coverage": coverage,
        "matched_ngrams": matched,
        "match_detected": match_detected,
        "corpus_status": "confirmed-match" if match_detected else "not-observable",
        "coverage_complete": False,
    }


def test_finalize_keeps_global_top_k_and_reassigns_ranks(tmp_path, monkeypatch):
    connection = connect(tmp_path / "manifest.sqlite")
    initialize(
        connection,
        metadata={
            "repo": "example/corpus",
            "revision": "abc",
            "ngram_size": "13",
            "ngram_coverage_threshold": "0.8",
            "candidates_per_item": "2",
            "evidence_schema_version": "3",
        },
        shards=[("a.zst", 1, 0), ("b.zst", 1, 0)],
    )
    shard_rows = {
        "a.zst": [_candidate("a-low", 0.4), _candidate("a-high", 0.9)],
        "b.zst": [_candidate("b-exact", 0.2, exact=True), _candidate("b-mid", 0.8)],
    }
    for worker_id in ("worker-a", "worker-b"):
        shard = claim_next(connection, worker_id=worker_id)
        output = tmp_path / f"{worker_id}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for row in shard_rows[shard["path"]]:
                handle.write(json.dumps(row) + "\n")
        assert mark_complete(
            connection,
            shard["path"],
            documents_scanned=10,
            evidence_rows=2,
            output_path=str(output),
            worker_id=worker_id,
            lease_token=shard["lease_token"],
        )

    item = SimpleNamespace(
        dataset=SimpleNamespace(value="humaneval"), item_id="item-1", prompt="prompt",
    )
    monkeypatch.setattr(MODULE, "benchmark_items", lambda: [item])
    destination = tmp_path / "final.jsonl"
    MODULE.finalize(SimpleNamespace(output=destination), connection)

    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [row["document_id"] for row in rows] == ["b-exact", "a-high"]
    assert [row["candidate_rank"] for row in rows] == [1, 2]
    assert {row["documents_scanned"] for row in rows} == {20}
    assert {row["corpus"] for row in rows} == {"example/corpus@abc"}
    assert {row["corpus_status"] for row in rows} == {"confirmed-match"}
    assert {row["coverage_complete"] for row in rows} == {True}


def test_finalize_converts_weak_partial_evidence_to_complete_no_match(tmp_path, monkeypatch):
    connection = connect(tmp_path / "manifest.sqlite")
    initialize(
        connection,
        metadata={
            "repo": "example/corpus",
            "revision": "abc",
            "ngram_size": "13",
            "ngram_coverage_threshold": "0.8",
            "candidates_per_item": "1",
            "evidence_schema_version": "3",
        },
        shards=[("a.zst", 1, 0)],
    )
    shard = claim_next(connection, worker_id="worker")
    output = tmp_path / "worker.jsonl"
    output.write_text(json.dumps(_candidate("weak", 0.4)) + "\n")
    assert mark_complete(
        connection,
        shard["path"],
        documents_scanned=10,
        evidence_rows=1,
        output_path=str(output),
        worker_id="worker",
        lease_token=shard["lease_token"],
    )
    item = SimpleNamespace(
        dataset=SimpleNamespace(value="humaneval"), item_id="item-1", prompt="prompt",
    )
    monkeypatch.setattr(MODULE, "benchmark_items", lambda: [item])
    destination = tmp_path / "final.jsonl"
    MODULE.finalize(SimpleNamespace(output=destination), connection)

    row = json.loads(destination.read_text())
    assert row["match_detected"] is False
    assert row["corpus_status"] == "no-match-found"
    assert row["coverage_complete"] is True
