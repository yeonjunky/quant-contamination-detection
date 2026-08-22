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
            finished_at TEXT
        );
        """
    )
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


def reset_running(connection: sqlite3.Connection) -> int:
    """Return work abandoned by a terminated process to the pending queue."""
    with connection:
        cursor = connection.execute(
            "UPDATE shards SET status='pending', error='interrupted', started_at=NULL "
            "WHERE status='running'"
        )
        return cursor.rowcount


def retry_failed(connection: sqlite3.Connection) -> int:
    """Schedule each currently failed shard for one new attempt."""
    with connection:
        cursor = connection.execute(
            "UPDATE shards SET status='pending', error=NULL WHERE status='failed'"
        )
        return cursor.rowcount


def claim_next(connection: sqlite3.Connection) -> dict[str, Any] | None:
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
            "started_at=CURRENT_TIMESTAMP, finished_at=NULL WHERE path=?",
            (row["path"],),
        )
        connection.commit()
        return dict(row)
    except BaseException:
        connection.rollback()
        raise


def mark_complete(
    connection: sqlite3.Connection,
    path: str,
    *,
    documents_scanned: int,
    evidence_rows: int,
    output_path: str,
) -> None:
    with connection:
        connection.execute(
            "UPDATE shards SET status='complete', documents_scanned=?, evidence_rows=?, "
            "output_path=?, finished_at=CURRENT_TIMESTAMP WHERE path=?",
            (documents_scanned, evidence_rows, output_path, path),
        )


def mark_failed(connection: sqlite3.Connection, path: str, error: str) -> None:
    with connection:
        connection.execute(
            "UPDATE shards SET status='failed', error=?, finished_at=CURRENT_TIMESTAMP WHERE path=?",
            (error[-4000:], path),
        )


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
