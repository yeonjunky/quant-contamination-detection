#!/usr/bin/env python3
"""Stream an Olmo training corpus and emit string-match evidence as JSONL."""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
from pathlib import Path

from qcd.data.humaneval import load_humaneval
from qcd.data.livecodebench import DEFAULT_RELEASE, load_livecodebench_split
from qcd.data.mbppplus import load_mbppplus
from qcd.ground_truth.string_match import MatchConfig, extract_text, scan_corpus


def _benchmark_items(name: str):
    if name == "humaneval":
        return load_humaneval()
    if name == "mbppplus":
        return load_mbppplus()
    pre, post = load_livecodebench_split(dt.datetime(2025, 1, 1), release_version=DEFAULT_RELEASE)
    return pre if name == "lcb_pre" else post


def _hf_documents(repo: str, revision: str, split: str, max_documents: int | None):
    from datasets import load_dataset

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
    parser.add_argument("--benchmark", choices=("humaneval", "mbppplus", "lcb_pre", "lcb_post"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf-repo")
    source.add_argument("--jsonl", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", help="HF commit; omitted resolves and records the current commit SHA")
    parser.add_argument("--stage", choices=("pretraining", "sft", "dpo", "rlvr"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ngram-size", type=int, default=13)
    parser.add_argument("--ngram-coverage-threshold", type=float, default=0.8)
    parser.add_argument("--max-documents", type=int, help="Smoke-test cap; omit for a complete stream scan")
    args = parser.parse_args()

    if args.hf_repo:
        from huggingface_hub import HfApi

        revision = args.revision or HfApi().dataset_info(args.hf_repo).sha
        documents = _hf_documents(args.hf_repo, revision, args.split, args.max_documents)
        corpus_name = f"{args.hf_repo}@{revision}"
    else:
        documents = _jsonl_documents(args.jsonl, args.max_documents)
        corpus_name = str(args.jsonl)
    rows = scan_corpus(
        _benchmark_items(args.benchmark), documents, corpus_name=corpus_name, stage=args.stage,
        config=MatchConfig(args.ngram_size, args.ngram_coverage_threshold),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
