#!/usr/bin/env python3
"""Stream an Olmo training corpus and emit string-match evidence as JSONL."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import itertools
import json
import sys
import time
from pathlib import Path

from qcd.data.humaneval import load_humaneval
from qcd.data.livecodebench import DEFAULT_RELEASE, load_livecodebench_split
from qcd.data.mbppplus import load_mbppplus
from qcd.ground_truth.string_match import MatchConfig, extract_text, scan_corpus


_BENCHMARKS = ("humaneval", "mbppplus", "lcb_pre", "lcb_post")


def _benchmark_items(name: str):
    if name == "all":
        pre, post = load_livecodebench_split(dt.datetime(2025, 1, 1), release_version=DEFAULT_RELEASE)
        items = [*load_humaneval(), *load_mbppplus(), *pre, *post]
        # LCB private tests are irrelevant to corpus matching and can occupy
        # several GB when retained across the full streaming scan.
        return [dataclasses.replace(item, metadata={}) for item in items]
    if name == "humaneval":
        return load_humaneval()
    if name == "mbppplus":
        return load_mbppplus()
    pre, post = load_livecodebench_split(dt.datetime(2025, 1, 1), release_version=DEFAULT_RELEASE)
    return pre if name == "lcb_pre" else post


def _hf_documents(
    repo: str,
    revision: str,
    split: str,
    max_documents: int | None,
    data_file: str | None = None,
):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_url

    if data_file:
        url = hf_hub_url(repo, data_file, repo_type="dataset", revision=revision)
        stream = load_dataset("json", data_files=url, split="train", streaming=True)
    else:
        stream = load_dataset(repo, revision=revision, split=split, streaming=True)
    rows = stream if max_documents is None else itertools.islice(stream, max_documents)
    for index, row in enumerate(rows):
        yield {"id": row.get("id", row.get("prompt_id", index)), "text": extract_text(row)}


def _jsonl_documents(path: Path, max_documents: int | None):
    with path.open(encoding="utf-8") as handle:
        rows = (json.loads(line) for line in handle if line.strip())
        for index, row in enumerate(itertools.islice(rows, max_documents)):
            yield {"id": row.get("id", index), "text": extract_text(row)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=(*_BENCHMARKS, "all"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf-repo")
    source.add_argument("--jsonl", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--hf-file",
        help="One file inside --hf-repo; avoids enumerating every shard and enables checkpointed scans",
    )
    parser.add_argument("--revision", help="HF commit; omitted resolves and records the current commit SHA")
    parser.add_argument("--stage", choices=("pretraining", "sft", "dpo", "rlvr"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ngram-size", type=int, default=13)
    parser.add_argument("--ngram-coverage-threshold", type=float, default=0.8)
    parser.add_argument("--max-documents", type=int, help="Smoke-test cap; omit for a complete stream scan")
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()

    if args.hf_file and not args.hf_repo:
        parser.error("--hf-file requires --hf-repo")

    if args.hf_repo:
        from huggingface_hub import HfApi

        revision = args.revision or HfApi().dataset_info(args.hf_repo).sha
        documents = _hf_documents(
            args.hf_repo, revision, args.split, args.max_documents, args.hf_file,
        )
        corpus_name = f"{args.hf_repo}@{revision}"
        if args.hf_file:
            corpus_name += f":{args.hf_file}"
    else:
        documents = _jsonl_documents(args.jsonl, args.max_documents)
        corpus_name = str(args.jsonl)
    started = time.monotonic()

    def report_progress(documents_scanned: int) -> None:
        elapsed = time.monotonic() - started
        rate = documents_scanned / elapsed if elapsed else 0.0
        print(
            f"scanned={documents_scanned:,} elapsed={elapsed:.1f}s rate={rate:.1f} docs/s",
            file=sys.stderr,
            flush=True,
        )

    rows = scan_corpus(
        _benchmark_items(args.benchmark), documents, corpus_name=corpus_name, stage=args.stage,
        config=MatchConfig(args.ngram_size, args.ngram_coverage_threshold),
        progress_every=args.progress_every,
        progress_callback=report_progress,
        coverage_complete=(args.max_documents is None and args.hf_file is None),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
