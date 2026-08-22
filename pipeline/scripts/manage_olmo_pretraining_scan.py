#!/usr/bin/env python3
"""Initialize, run, inspect, and finalize a resumable Olmo pretraining scan."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

from qcd.data.humaneval import load_humaneval
from qcd.data.livecodebench import DEFAULT_RELEASE, load_livecodebench_split
from qcd.data.mbppplus import load_mbppplus
from qcd.ground_truth.shard_manifest import (
    claim_next,
    connect,
    heartbeat,
    initialize,
    mark_complete,
    mark_failed,
    recover_stale,
    require_metadata,
    retry_failed,
    summary,
)
from qcd.ground_truth.string_match import MatchConfig, extract_text, scan_corpus, tokenize


EVIDENCE_SCHEMA_VERSION = "2"


def retrieval_metadata(args) -> dict[str, str]:
    return {
        "ngram_size": str(args.ngram_size),
        "ngram_coverage_threshold": str(args.ngram_coverage_threshold),
        "candidates_per_item": str(args.candidates_per_item),
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
    }


def benchmark_items():
    pre, post = load_livecodebench_split(
        dt.datetime(2025, 1, 1), release_version=DEFAULT_RELEASE,
    )
    items = [*load_humaneval(), *load_mbppplus(), *pre, *post]
    return [dataclasses.replace(item, metadata={}) for item in items]


def list_shards(repo: str, revision: str):
    """Read the exact revision manifest, excluding files removed from that commit."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError(f"Hub resolved {revision!r} to unexpected commit {info.sha!r}")
    for entry in info.siblings:
        if entry.rfilename.endswith(".jsonl.zst"):
            priority = 0 if "software" in entry.rfilename.casefold() else 1
            yield entry.rfilename, int(entry.size or 0), priority


def documents(repo: str, revision: str, shard: str):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo, shard, repo_type="dataset", revision=revision)
    stream = load_dataset("json", data_files=url, split="train", streaming=True)
    for index, row in enumerate(stream):
        yield {
            "id": row.get("id", index),
            "text": extract_text(row),
            "source": row.get("source"),
            "version": row.get("version"),
            "created": row.get("created"),
            "added": row.get("added"),
        }


def output_name(shard: str) -> str:
    digest = hashlib.sha256(shard.encode()).hexdigest()[:12]
    return f"{Path(shard).stem.removesuffix('.jsonl')}-{digest}.jsonl"


def write_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args, connection) -> None:
    if args.retry_failed:
        retried = retry_failed(connection)
        print(f"returned {retried} failed shard(s) to pending", file=sys.stderr)
    items = benchmark_items()
    processed = 0
    while args.max_shards is None or processed < args.max_shards:
        recovered = recover_stale(
            connection, stale_after_seconds=args.stale_after_seconds,
        )
        if recovered:
            print(f"worker={args.worker_id} recovered={recovered}", file=sys.stderr, flush=True)
        shard = claim_next(connection, worker_id=args.worker_id)
        if shard is None:
            break
        path = shard["path"]
        destination = args.output_dir / args.worker_id / output_name(path)
        started = time.monotonic()

        def progress(count: int) -> None:
            if not heartbeat(connection, path, worker_id=args.worker_id):
                raise RuntimeError(f"worker lease lost for {path}")
            elapsed = time.monotonic() - started
            print(
                f"worker={args.worker_id} shard={path} documents={count:,} "
                f"rate={count / elapsed:.1f}/s",
                file=sys.stderr, flush=True,
            )

        try:
            completion_count = []
            rows = scan_corpus(
                items, documents(args.repo, args.revision, path),
                corpus_name=f"{args.repo}@{args.revision}:{path}", stage="pretraining",
                config=MatchConfig(args.ngram_size, args.ngram_coverage_threshold),
                progress_every=args.progress_every, progress_callback=progress,
                completion_callback=completion_count.append,
                top_k=args.candidates_per_item, evidence_only=True,
                include_document_text=True,
            )
            document_count = completion_count[0]
            evidence = rows
            if not heartbeat(connection, path, worker_id=args.worker_id):
                raise RuntimeError(f"worker lease lost before writing {path}")
            write_atomic(destination, evidence)
            if not mark_complete(
                connection, path, documents_scanned=document_count,
                evidence_rows=len(evidence), output_path=str(destination),
                worker_id=args.worker_id,
            ):
                raise RuntimeError(f"worker lease lost before completing {path}")
            processed += 1
            print(
                f"worker={args.worker_id} complete shard={path} documents={document_count:,} "
                f"evidence={len(evidence):,}", file=sys.stderr, flush=True,
            )
        except BaseException as error:
            mark_failed(
                connection, path, f"{type(error).__name__}: {error}",
                worker_id=args.worker_id,
            )
            print(
                f"worker={args.worker_id} failed shard={path}: {error}",
                file=sys.stderr, flush=True,
            )
            if not args.keep_going:
                raise


def print_status(connection) -> None:
    stats = summary(connection)
    total = stats["total"]
    completed = stats.get("complete", 0)
    payload = {
        **stats,
        "completion_fraction": completed / total if total else 0.0,
        "byte_fraction": (
            stats["completed_bytes"] / stats["compressed_bytes"]
            if stats["compressed_bytes"] else 0.0
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def finalize(args, connection) -> None:
    stats = summary(connection)
    if stats.get("complete", 0) != stats["total"]:
        raise RuntimeError(
            f"cannot finalize an incomplete scan: {stats.get('complete', 0):,}/"
            f"{stats['total']:,} shards complete"
        )
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    candidates_per_item = int(metadata["candidates_per_item"])
    ngram_size = int(metadata["ngram_size"])
    ngram_coverage_threshold = float(metadata["ngram_coverage_threshold"])
    best: dict[tuple[str, str], list[dict]] = {}
    for record in connection.execute(
        "SELECT path, output_path FROM shards WHERE status='complete' ORDER BY path"
    ):
        with Path(record["output_path"]).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["dataset"], row["item_id"])
                row["source_shard"] = record["path"]
                candidates = best.setdefault(key, [])
                candidates.append(row)
                candidates.sort(key=_final_evidence_rank, reverse=True)
                del candidates[candidates_per_item:]

    corpus = f"{metadata['repo']}@{metadata['revision']}"
    rows = []
    for item in benchmark_items():
        key = (item.dataset.value, item.item_id)
        item_candidates = best.get(key, [])
        if not item_candidates:
            item_candidates = [{
                "item_id": item.item_id,
                "dataset": item.dataset.value,
                "stage": "pretraining",
                "method_family": "instance_string",
                "normalized_verbatim": False,
                "ngram_size": ngram_size,
                "ngram_coverage": 0.0,
                "ngram_threshold": ngram_coverage_threshold,
                "string_match_label": False,
                "document_id": None,
                "matched_ngrams": 0,
                "query_ngrams": max(0, len(tokenize(item.prompt)) - ngram_size + 1),
                "source_shard": None,
            }]
        for candidate_rank, row in enumerate(item_candidates, start=1):
            row["candidate_rank"] = candidate_rank
            row["corpus"] = corpus
            row["documents_scanned"] = stats["documents_scanned"]
            rows.append(row)
    write_atomic(args.output, rows)
    print(f"wrote {len(rows):,} item rows to {args.output}")


def _final_evidence_rank(row: dict) -> tuple:
    """Stable global ordering for candidates collected from different shards."""
    return (
        bool(row["normalized_verbatim"]),
        float(row["ngram_coverage"]),
        int(row["matched_ngrams"]),
        str(row["source_shard"]),
        str(row["document_id"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--repo", required=True)
    init_parser.add_argument("--revision")
    init_parser.add_argument("--ngram-size", type=int, default=13)
    init_parser.add_argument("--ngram-coverage-threshold", type=float, default=0.8)
    init_parser.add_argument("--candidates-per-item", type=int, default=5)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--repo", required=True)
    run_parser.add_argument("--revision", required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--worker-id", default=f"worker-{uuid.uuid4().hex[:12]}")
    run_parser.add_argument("--stale-after-seconds", type=int, default=600)
    run_parser.add_argument("--max-shards", type=int)
    run_parser.add_argument("--retry-failed", action="store_true")
    run_parser.add_argument("--keep-going", action="store_true")
    run_parser.add_argument("--progress-every", type=int, default=10_000)
    run_parser.add_argument("--ngram-size", type=int, default=13)
    run_parser.add_argument("--ngram-coverage-threshold", type=float, default=0.8)
    run_parser.add_argument("--candidates-per-item", type=int, default=5)

    subparsers.add_parser("status")
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    connection = connect(args.manifest)

    if args.command == "init":
        from huggingface_hub import HfApi

        revision = args.revision or HfApi().dataset_info(args.repo).sha
        count = initialize(
            connection,
            metadata={
                "repo": args.repo, "revision": revision, **retrieval_metadata(args),
            },
            shards=list_shards(args.repo, revision),
        )
        print(f"added {count:,} shard(s); revision={revision}")
        print_status(connection)
    elif args.command == "run":
        try:
            require_metadata(
                connection,
                {"repo": args.repo, "revision": args.revision, **retrieval_metadata(args)},
            )
        except ValueError as error:
            parser.error(str(error))
        run(args, connection)
        print_status(connection)
    elif args.command == "finalize":
        finalize(args, connection)
    else:
        print_status(connection)


if __name__ == "__main__":
    main()
