"""Transactional shard bookkeeping for resumable corpus scans."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS shards (
            path TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            priority INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'running', 'complete', 'failed')),
            attempts INTEGER NOT NULL DEFAULT 0,
            documents_scanned INTEGER,
            evidence_rows INTEGER,
            output_path TEXT,
            error TEXT,
            started_at TEXT,
            heartbeat_at TEXT,
            worker_id TEXT,
            finished_at TEXT
        );
        """
    )
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(shards)")}
    for name in ("heartbeat_at", "worker_id"):
        if name not in columns:
            connection.execute(f"ALTER TABLE shards ADD COLUMN {name} TEXT")
    connection.commit()
    return connection


def initialize(
    connection: sqlite3.Connection,
    *,
    metadata: dict[str, str],
    shards: Iterable[tuple[str, int, int]],
) -> int:
    """Insert an immutable corpus identity and any previously unseen shards."""
    with connection:
        existing = dict(connection.execute("SELECT key, value FROM metadata"))
        for key, value in metadata.items():
            if key in existing and existing[key] != value:
                raise ValueError(f"manifest {key} is {existing[key]!r}, not {value!r}")
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)", (key, value),
            )
        before = connection.total_changes
        connection.executemany(
            "INSERT OR IGNORE INTO shards(path, size_bytes, priority) VALUES (?, ?, ?)", shards,
        )
        return connection.total_changes - before


def require_metadata(connection: sqlite3.Connection, expected: dict[str, str]) -> None:
    """Reject workers whose retrieval configuration differs from the manifest."""
    actual = dict(connection.execute("SELECT key, value FROM metadata"))
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={observed!r} (expected {wanted!r})"
            for key, (observed, wanted) in sorted(mismatches.items())
        )
        raise ValueError(f"manifest metadata mismatch: {details}")


def recover_stale(connection: sqlite3.Connection, *, stale_after_seconds: int) -> int:
    """Requeue only leases whose owner stopped sending heartbeats."""
    if stale_after_seconds < 1:
        raise ValueError("stale_after_seconds must be positive")
    with connection:
        cursor = connection.execute(
            "UPDATE shards SET status='pending', error='stale worker lease recovered', "
            "started_at=NULL, heartbeat_at=NULL, worker_id=NULL "
            "WHERE status='running' AND (heartbeat_at IS NULL OR "
            "heartbeat_at < datetime('now', ?))",
            (f"-{stale_after_seconds} seconds",),
        )
        return cursor.rowcount


def retry_failed(connection: sqlite3.Connection) -> int:
    """Schedule each currently failed shard for one new attempt."""
    with connection:
        cursor = connection.execute(
            "UPDATE shards SET status='pending', error=NULL WHERE status='failed'"
        )
        return cursor.rowcount


def invalidate_completed(connection: sqlite3.Connection, *, reason: str) -> int:
    """Requeue completed shards after an evidence-schema change."""
    if not reason.strip():
        raise ValueError("reason must be non-empty")
    with connection:
        cursor = connection.execute(
            "UPDATE shards SET status='pending', documents_scanned=NULL, evidence_rows=NULL, "
            "output_path=NULL, error=?, started_at=NULL, heartbeat_at=NULL, worker_id=NULL, "
            "finished_at=NULL WHERE status='complete'",
            (f"invalidated: {reason}",),
        )
        return cursor.rowcount


def claim_next(connection: sqlite3.Connection, *, worker_id: str) -> dict[str, Any] | None:
    if not worker_id.strip():
        raise ValueError("worker_id must be non-empty")
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(
            "SELECT * FROM shards WHERE status='pending' "
            "ORDER BY priority, attempts, path LIMIT 1",
        ).fetchone()
        if row is None:
            connection.commit()
            return None
        connection.execute(
            "UPDATE shards SET status='running', attempts=attempts+1, error=NULL, "
            "started_at=CURRENT_TIMESTAMP, heartbeat_at=CURRENT_TIMESTAMP, worker_id=?, "
            "finished_at=NULL WHERE path=? AND status='pending'",
            (worker_id, row["path"]),
        )
        connection.commit()
        return dict(row)
    except BaseException:
        connection.rollback()
        raise


def heartbeat(connection: sqlite3.Connection, path: str, *, worker_id: str) -> bool:
    with connection:
        cursor = connection.execute(
            "UPDATE shards SET heartbeat_at=CURRENT_TIMESTAMP "
            "WHERE path=? AND status='running' AND worker_id=?",
            (path, worker_id),
        )
        return cursor.rowcount == 1


def mark_complete(
    connection: sqlite3.Connection,
    path: str,
    *,
    documents_scanned: int,
    evidence_rows: int,
    output_path: str,
    worker_id: str,
) -> bool:
    with connection:
        cursor = connection.execute(
            "UPDATE shards SET status='complete', documents_scanned=?, evidence_rows=?, "
            "output_path=?, heartbeat_at=CURRENT_TIMESTAMP, finished_at=CURRENT_TIMESTAMP "
            "WHERE path=? AND status='running' AND worker_id=?",
            (documents_scanned, evidence_rows, output_path, path, worker_id),
        )
        return cursor.rowcount == 1


def mark_failed(connection: sqlite3.Connection, path: str, error: str, *, worker_id: str) -> bool:
    with connection:
        cursor = connection.execute(
            "UPDATE shards SET status='failed', error=?, heartbeat_at=CURRENT_TIMESTAMP, "
            "finished_at=CURRENT_TIMESTAMP "
            "WHERE path=? AND status='running' AND worker_id=?",
            (error[-4000:], path, worker_id),
        )
        return cursor.rowcount == 1


def summary(connection: sqlite3.Connection) -> dict[str, int]:
    counts = {row["status"]: row["count"] for row in connection.execute(
        "SELECT status, COUNT(*) AS count FROM shards GROUP BY status"
    )}
    counts["total"] = sum(counts.values())
    counts["compressed_bytes"] = connection.execute(
        "SELECT COALESCE(SUM(size_bytes), 0) FROM shards"
    ).fetchone()[0]
    counts["completed_bytes"] = connection.execute(
        "SELECT COALESCE(SUM(size_bytes), 0) FROM shards WHERE status='complete'"
    ).fetchone()[0]
    counts["documents_scanned"] = connection.execute(
        "SELECT COALESCE(SUM(documents_scanned), 0) FROM shards WHERE status='complete'"
    ).fetchone()[0]
    return counts
