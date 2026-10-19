"""CPU-only first-pass verbatim and token n-gram corpus search.

This module deliberately does not call its output semantic ground truth.  It
implements family (i) of paper section 5 only: auditable retrieval evidence
that can seed the later AST/semantic/paraphrase and TRACER stages.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from qcd.data.schema import CorpusEvidenceFamily, CorpusReferenceStatus, Item

#: Paper §5 step 5's family (i). Taken from the schema enum so the on-disk
#: spelling cannot drift from `data/schema.py`'s.
METHOD_FAMILY = CorpusEvidenceFamily.INSTANCE_STRING.value

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

    def matches(self, coverage: float) -> bool:
        return coverage >= self.ngram_coverage_threshold


#: The frozen `confirmed-match` rule (`OLMO_GROUND_TRUTH.md`, 2026-10-19).
PRIMARY_CONFIG = MatchConfig(13, 0.8)
#: The sensitivity rule reported beside it: the 8-gram / 0.5-overlap rule the
#: Olmo 3 report gives for its own post-training decontamination
#: (arXiv:2512.13961). It never replaces the primary status.
SENSITIVITY_CONFIG = MatchConfig(8, 0.5)


def _ngrams(tokens: tuple[str, ...], n: int) -> set[tuple[str, ...]]:
    return {tokens[i : i + n] for i in range(max(0, len(tokens) - n + 1))}


def scan_corpus(
    items: Iterable[Item],
    documents: Iterable[Mapping[str, Any]],
    *,
    model: str,
    corpus_name: str,
    stage: str,
    config: MatchConfig = MatchConfig(),
    sensitivity: MatchConfig | None = None,
    progress_every: int | None = None,
    progress_callback: Callable[[int], None] | None = None,
    completion_callback: Callable[[int], None] | None = None,
    top_k: int = 1,
    evidence_only: bool = False,
    include_document_text: bool = False,
    coverage_complete: bool = False,
) -> list[dict[str, Any]]:
    """Scan an iterable once and return bounded evidence rows per item.

    ``model`` is the `models/registry.py` checkpoint name this corpus belongs
    to, and it is written into every row.  Paper §4.2 stores corpus evidence
    per model, and the two Olmo arms do not share a pretraining mix
    (`ground_truth/olmo_corpora.py`), so an unattributed row cannot be turned
    back into a per-model status later.

    Documents require ``text`` and may provide ``id``.  The algorithm indexes
    benchmark n-grams, not corpus documents, so memory is bounded by the small
    benchmark side.  It is suitable for HF ``IterableDataset`` streams, though
    a full multi-terabyte pretraining pass remains an operationally expensive
    fallback rather than a substitute for a public/persistent corpus index.
    Callers must set ``coverage_complete=True`` only when the requested corpus
    scope was exhausted; otherwise a non-match remains ``not-observable``.

    With ``sensitivity`` set, the same pass also counts that rule's n-grams: a
    document sharing an n-gram of either width becomes a candidate, every row
    carries the sensitivity coverage and a separate ``sensitivity_*`` status,
    and candidates are ranked primary-match first, then sensitivity-match, so
    a sensitivity-only match is not pushed out of the ``top_k`` bound by
    lower-coverage candidates.  The primary fields are unchanged by it.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty registry model name")
    output_items: list[tuple[tuple[str, str], str, str]] = []
    queries: dict[tuple[str, str], dict[str, Any]] = {}
    inverted: dict[tuple[str, ...], set[tuple[str, str]]] = defaultdict(set)
    sensitivity_inverted: dict[tuple[str, ...], set[tuple[str, str]]] = defaultdict(set)
    for item in items:
        query_key = (item.dataset.value, item.item_id)
        normalized = normalize_text(item.prompt)
        tokens = tokenize(item.prompt)
        grams = _ngrams(tokens, config.ngram_size)
        sensitivity_grams = _ngrams(tokens, sensitivity.ngram_size) if sensitivity else set()
        output_items.append((query_key, item.item_id, item.dataset.value))
        queries[query_key] = {
            "normalized": normalized,
            "grams": grams,
            "sensitivity_grams": sensitivity_grams,
        }
        for gram in grams:
            inverted[gram].add(query_key)
        for gram in sensitivity_grams:
            sensitivity_inverted[gram].add(query_key)
    # LiveCodeBench items carry very large private-test metadata.  Retain only
    # the lightweight query/output fields above during a multi-million-row
    # corpus scan.
    del items
    short_queries = tuple(
        (query_key, query["normalized"])
        for query_key, query in queries.items()
        if not query["grams"]
    )

    best: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    docs_scanned = 0
    for doc_index, document in enumerate(documents):
        docs_scanned += 1
        text = str(document.get("text", ""))
        normalized_doc = normalize_text(text)
        doc_tokens = tokenize(text)
        doc_grams = _ngrams(doc_tokens, config.ngram_size)
        counts: dict[tuple[str, str], int] = defaultdict(int)
        example_gram: dict[tuple[str, str], tuple[str, ...]] = {}
        for gram in doc_grams:
            for query_key in inverted.get(gram, ()):
                counts[query_key] += 1
                example_gram.setdefault(query_key, gram)
        sensitivity_counts: dict[tuple[str, str], int] = defaultdict(int)
        if sensitivity:
            for gram in _ngrams(doc_tokens, sensitivity.ngram_size):
                for query_key in sensitivity_inverted.get(gram, ()):
                    sensitivity_counts[query_key] += 1
                    example_gram.setdefault(query_key, gram)

        candidates = set(counts) | set(sensitivity_counts)
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
            gram = example_gram.get(query_key)
            token_start = _find_subsequence(doc_tokens, gram) if gram else None
            context = None
            if token_start is not None:
                left = max(0, token_start - 30)
                right = min(len(doc_tokens), token_start + len(gram) + 30)
                context = " ".join(doc_tokens[left:right])
            evidence = {
                    "document_id": str(document.get("id", doc_index)),
                    "normalized_verbatim": exact,
                    "ngram_coverage": coverage,
                    "matched_ngrams": counts[query_key],
                    "query_ngrams": total,
                    "exact_char_start": normalized_doc.find(query["normalized"]) if exact else None,
                    "matched_token_start": token_start,
                    "matched_ngram_example": " ".join(gram) if gram else None,
                    "match_context": context,
                }
            if sensitivity:
                sensitivity_total = len(query["sensitivity_grams"])
                evidence.update({
                    "sensitivity_ngram_coverage": (
                        sensitivity_counts[query_key] / sensitivity_total
                        if sensitivity_total else 0.0
                    ),
                    "sensitivity_matched_ngrams": sensitivity_counts[query_key],
                    "sensitivity_query_ngrams": sensitivity_total,
                })
            if include_document_text:
                evidence.update(
                    {
                        "document_text": text,
                        "normalized_document_text": normalized_doc,
                        "document_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "document_text_truncated": False,
                        "document_source": document.get("source"),
                        "document_version": document.get("version"),
                        "document_created": document.get("created"),
                        "document_added": document.get("added"),
                    }
                )
            candidates_for_query = best[query_key]
            candidates_for_query.append(evidence)
            candidates_for_query.sort(
                key=lambda row: evidence_rank(row, config, sensitivity), reverse=True
            )
            del candidates_for_query[top_k:]
        if progress_every and docs_scanned % progress_every == 0 and progress_callback:
            progress_callback(docs_scanned)

    if completion_callback:
        completion_callback(docs_scanned)
    rows: list[dict[str, Any]] = []
    for query_key, item_id, dataset in output_items:
        evidence_rows = best.get(query_key, [])
        if not evidence_rows and evidence_only:
            continue
        for candidate_rank, evidence in enumerate(evidence_rows or [{}], start=1):
            coverage = float(evidence.get("ngram_coverage", 0.0))
            exact = bool(evidence.get("normalized_verbatim", False))
            matched = exact or config.matches(coverage)
            status = _status(matched, coverage_complete)
            row = {
                "item_id": item_id,
                "dataset": dataset,
                "model": model,
                "corpus": corpus_name,
                "stage": stage,
                "method_family": METHOD_FAMILY,
                "normalized_verbatim": exact,
                "ngram_size": config.ngram_size,
                "ngram_coverage": coverage,
                "ngram_threshold": config.ngram_coverage_threshold,
                "match_detected": matched,
                "corpus_status": status.value,
                "coverage_complete": coverage_complete,
                "document_id": evidence.get("document_id"),
                "matched_ngrams": int(evidence.get("matched_ngrams", 0)),
                "query_ngrams": int(evidence.get("query_ngrams", len(queries[query_key]["grams"]))),
                "documents_scanned": docs_scanned,
                "candidate_rank": candidate_rank,
            }
            if sensitivity:
                sensitivity_coverage = float(evidence.get("sensitivity_ngram_coverage", 0.0))
                sensitivity_matched = exact or sensitivity.matches(sensitivity_coverage)
                row.update({
                    "sensitivity_ngram_size": sensitivity.ngram_size,
                    "sensitivity_ngram_coverage": sensitivity_coverage,
                    "sensitivity_ngram_threshold": sensitivity.ngram_coverage_threshold,
                    "sensitivity_match_detected": sensitivity_matched,
                    "sensitivity_corpus_status": _status(
                        sensitivity_matched, coverage_complete
                    ).value,
                    "sensitivity_matched_ngrams": int(evidence.get("sensitivity_matched_ngrams", 0)),
                    "sensitivity_query_ngrams": int(evidence.get(
                        "sensitivity_query_ngrams", len(queries[query_key]["sensitivity_grams"])
                    )),
                })
            row.update({key: value for key, value in evidence.items() if key not in row})
            rows.append(row)
    return rows


def _status(matched: bool, coverage_complete: bool) -> CorpusReferenceStatus:
    if matched:
        return CorpusReferenceStatus.CONFIRMED_MATCH
    if coverage_complete:
        return CorpusReferenceStatus.NO_MATCH_FOUND
    return CorpusReferenceStatus.NOT_OBSERVABLE


def evidence_rank(
    evidence: Mapping[str, Any],
    config: MatchConfig,
    sensitivity: MatchConfig | None = None,
) -> tuple[Any, ...]:
    """Candidate order: verbatim, primary match, sensitivity match, then
    coverage.  Without ``sensitivity`` this is the verbatim/coverage order
    alone, because a primary match is monotone in primary coverage."""
    exact = bool(evidence["normalized_verbatim"])
    coverage = float(evidence["ngram_coverage"])
    sensitivity_coverage = float(evidence.get("sensitivity_ngram_coverage", 0.0))
    return (
        exact,
        config.matches(coverage),
        bool(sensitivity and sensitivity.matches(sensitivity_coverage)),
        coverage,
        sensitivity_coverage,
        int(evidence["matched_ngrams"]),
        str(evidence["document_id"]),
    )


def _find_subsequence(tokens: tuple[str, ...], needle: tuple[str, ...] | None) -> int | None:
    if not needle:
        return None
    width = len(needle)
    for index in range(len(tokens) - width + 1):
        if tokens[index : index + width] == needle:
            return index
    return None
