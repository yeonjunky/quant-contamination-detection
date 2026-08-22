"""CPU-only first-pass verbatim and token n-gram corpus search.

This module deliberately does not call its output semantic ground truth.  It
implements family (i) of paper section 5 only: auditable retrieval evidence
that can seed the later AST/semantic/paraphrase and TRACER stages.
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from qcd.data.schema import Item

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[^\w\s]", re.UNICODE)


def normalize_text(text: str) -> str:
    """NFKC-normalize, lowercase, and collapse whitespace for matching."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(normalize_text(text)))


def extract_text(value: Any) -> str:
    """Flatten text-bearing fields without mixing IDs/source metadata into evidence."""
    parts: list[str] = []

    top_level_text_keys = {
        "text", "prompt", "solution", "ground_truth", "messages", "chosen",
        "rejected", "source_prompt", "outputs",
    }
    nested_text_keys = {"content", "reasoning_content", "text"}

    def visit(node: Any, *, nested: bool = False) -> None:
        if isinstance(node, str):
            parts.append(node)
        elif isinstance(node, Mapping):
            allowed = nested_text_keys if nested else top_level_text_keys
            for key, child in node.items():
                if key in allowed:
                    visit(child, nested=True)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child, nested=nested)

    visit(value)
    return "\n".join(parts)


@dataclasses.dataclass(frozen=True)
class MatchConfig:
    ngram_size: int = 13
    ngram_coverage_threshold: float = 0.8

    def __post_init__(self) -> None:
        if self.ngram_size < 1:
            raise ValueError("ngram_size must be positive")
        if not 0.0 <= self.ngram_coverage_threshold <= 1.0:
            raise ValueError("ngram_coverage_threshold must be in [0, 1]")


def _ngrams(tokens: tuple[str, ...], n: int) -> set[tuple[str, ...]]:
    return {tokens[i : i + n] for i in range(max(0, len(tokens) - n + 1))}


def scan_corpus(
    items: Iterable[Item],
    documents: Iterable[Mapping[str, Any]],
    *,
    corpus_name: str,
    stage: str,
    config: MatchConfig = MatchConfig(),
    progress_every: int | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    """Scan an iterable once and return one best-evidence row per item.

    Documents require ``text`` and may provide ``id``.  The algorithm indexes
    benchmark n-grams, not corpus documents, so memory is bounded by the small
    benchmark side.  It is suitable for HF ``IterableDataset`` streams, though
    a full multi-terabyte pretraining pass remains an operationally expensive
    fallback rather than a substitute for a public/persistent corpus index.
    """
    output_items: list[tuple[tuple[str, str], str, str]] = []
    queries: dict[tuple[str, str], dict[str, Any]] = {}
    inverted: dict[tuple[str, ...], set[tuple[str, str]]] = defaultdict(set)
    for item in items:
        query_key = (item.dataset.value, item.item_id)
        normalized = normalize_text(item.prompt)
        grams = _ngrams(tokenize(item.prompt), config.ngram_size)
        output_items.append((query_key, item.item_id, item.dataset.value))
        queries[query_key] = {"normalized": normalized, "grams": grams}
        for gram in grams:
            inverted[gram].add(query_key)
    # LiveCodeBench items carry very large private-test metadata.  Retain only
    # the lightweight query/output fields above during a multi-million-row
    # corpus scan.
    del items
    short_queries = tuple(
        (query_key, query["normalized"])
        for query_key, query in queries.items()
        if not query["grams"]
    )

    best: dict[tuple[str, str], dict[str, Any]] = {}
    docs_scanned = 0
    for doc_index, document in enumerate(documents):
        docs_scanned += 1
        text = str(document.get("text", ""))
        normalized_doc = normalize_text(text)
        doc_grams = _ngrams(tokenize(text), config.ngram_size)
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for gram in doc_grams:
            for query_key in inverted.get(gram, ()):
                counts[query_key] += 1

        candidates = set(counts)
        # Short prompts have no n-grams at the configured width but still get
        # a normalized-verbatim check.
        # Any query long enough to have n-grams must share at least one of them
        # with a verbatim-containing document.  Only short queries need this
        # direct fallback; checking every benchmark prompt in every corpus
        # document would make a billion-document scan infeasible.
        candidates.update(
            query_key
            for query_key, normalized_query in short_queries
            if normalized_query in normalized_doc
        )
        for query_key in candidates:
            query = queries[query_key]
            total = len(query["grams"])
            coverage = counts[query_key] / total if total else 0.0
            exact = bool(query["normalized"] and query["normalized"] in normalized_doc)
            previous = best.get(query_key)
            if previous is None or (exact, coverage) > (previous["normalized_verbatim"], previous["ngram_coverage"]):
                best[query_key] = {
                    "document_id": str(document.get("id", doc_index)),
                    "normalized_verbatim": exact,
                    "ngram_coverage": coverage,
                    "matched_ngrams": counts[query_key],
                    "query_ngrams": total,
                }
        if progress_every and docs_scanned % progress_every == 0 and progress_callback:
            progress_callback(docs_scanned)

    rows: list[dict[str, Any]] = []
    for query_key, item_id, dataset in output_items:
        evidence = best.get(query_key, {})
        coverage = float(evidence.get("ngram_coverage", 0.0))
        exact = bool(evidence.get("normalized_verbatim", False))
        rows.append(
            {
                "item_id": item_id,
                "dataset": dataset,
                "corpus": corpus_name,
                "stage": stage,
                "method_family": "instance_string",
                "normalized_verbatim": exact,
                "ngram_size": config.ngram_size,
                "ngram_coverage": coverage,
                "ngram_threshold": config.ngram_coverage_threshold,
                "string_match_label": exact or coverage >= config.ngram_coverage_threshold,
                "document_id": evidence.get("document_id"),
                "matched_ngrams": int(evidence.get("matched_ngrams", 0)),
                "query_ngrams": int(evidence.get("query_ngrams", len(queries[query_key]["grams"]))),
                "documents_scanned": docs_scanned,
            }
        )
    return rows
