from qcd.ground_truth.shard_manifest import (
    claim_next,
    connect,
    initialize,
    mark_complete,
    mark_failed,
    reset_running,
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

    first = claim_next(connection)
    assert first["path"] == "software.jsonl.zst"
    assert reset_running(connection) == 1
    assert claim_next(connection)["path"] == "software.jsonl.zst"
    mark_complete(
        connection, "software.jsonl.zst", documents_scanned=7,
        evidence_rows=2, output_path="software.jsonl",
    )

    assert claim_next(connection)["path"] == "general.jsonl.zst"
    mark_failed(connection, "general.jsonl.zst", "network error")
    assert claim_next(connection) is None
    assert retry_failed(connection) == 1
    assert claim_next(connection)["path"] == "general.jsonl.zst"

    stats = summary(connection)
    assert stats["total"] == 2
    assert stats["complete"] == 1
    assert stats["running"] == 1
    assert stats["compressed_bytes"] == 30
    assert stats["completed_bytes"] == 10
    assert stats["documents_scanned"] == 7


def test_manifest_rejects_a_different_revision(tmp_path):
    connection = connect(tmp_path / "manifest.sqlite")
    initialize(connection, metadata={"revision": "abc"}, shards=[])

    try:
        initialize(connection, metadata={"revision": "different"}, shards=[])
    except ValueError as error:
        assert "revision" in str(error)
    else:
        raise AssertionError("revision mismatch was accepted")
