from qcd.ground_truth.shard_manifest import (
    claim_next,
    connect,
    heartbeat,
    initialize,
    mark_complete,
    mark_failed,
    recover_stale,
    retry_failed,
    summary,
)


def test_manifest_resumes_and_prioritizes_shards(tmp_path):
    connection = connect(tmp_path / "manifest.sqlite")
    initialize(
        connection,
        metadata={"repo": "example/corpus", "revision": "abc"},
        shards=[("general.jsonl.zst", 20, 1), ("software.jsonl.zst", 10, 0)],
    )

    first = claim_next(connection, worker_id="worker-a")
    assert first["path"] == "software.jsonl.zst"
    assert recover_stale(connection, stale_after_seconds=600) == 0
    connection.execute(
        "UPDATE shards SET heartbeat_at=datetime('now', '-601 seconds') "
        "WHERE path='software.jsonl.zst'"
    )
    connection.commit()
    assert recover_stale(connection, stale_after_seconds=600) == 1
    assert claim_next(connection, worker_id="worker-b")["path"] == "software.jsonl.zst"
    mark_complete(
        connection, "software.jsonl.zst", documents_scanned=7,
        evidence_rows=2, output_path="software.jsonl", worker_id="worker-b",
    )

    assert claim_next(connection, worker_id="worker-a")["path"] == "general.jsonl.zst"
    mark_failed(connection, "general.jsonl.zst", "network error", worker_id="worker-a")
    assert claim_next(connection, worker_id="worker-a") is None
    assert retry_failed(connection) == 1
    assert claim_next(connection, worker_id="worker-a")["path"] == "general.jsonl.zst"

    stats = summary(connection)
    assert stats["total"] == 2
    assert stats["complete"] == 1
    assert stats["running"] == 1
    assert stats["compressed_bytes"] == 30
    assert stats["completed_bytes"] == 10
    assert stats["documents_scanned"] == 7


def test_worker_ownership_prevents_stale_completion(tmp_path):
    path = tmp_path / "manifest.sqlite"
    first = connect(path)
    second = connect(path)
    initialize(first, metadata={"revision": "abc"}, shards=[("one", 1, 0)])
    claim_next(first, worker_id="old-worker")
    first.execute("UPDATE shards SET heartbeat_at=datetime('now', '-601 seconds')")
    first.commit()
    assert recover_stale(second, stale_after_seconds=600) == 1
    claim_next(second, worker_id="new-worker")

    assert heartbeat(first, "one", worker_id="old-worker") is False
    assert mark_complete(
        first, "one", documents_scanned=1, evidence_rows=0,
        output_path="old.jsonl", worker_id="old-worker",
    ) is False
    assert mark_failed(first, "one", "late", worker_id="old-worker") is False
    assert mark_complete(
        second, "one", documents_scanned=2, evidence_rows=1,
        output_path="new.jsonl", worker_id="new-worker",
    ) is True
    row = second.execute("SELECT status, output_path FROM shards WHERE path='one'").fetchone()
    assert tuple(row) == ("complete", "new.jsonl")


def test_manifest_rejects_a_different_revision(tmp_path):
    connection = connect(tmp_path / "manifest.sqlite")
    initialize(connection, metadata={"revision": "abc"}, shards=[])

    try:
        initialize(connection, metadata={"revision": "different"}, shards=[])
    except ValueError as error:
        assert "revision" in str(error)
    else:
        raise AssertionError("revision mismatch was accepted")
