"""CDD (Contamination Detection via output Distribution) — the peakedness-
based detector characterized in paper §2.4, introduced by Dong et al.
(2024), replicated in arXiv:2603.03203 (Sela).

Formula pinned in constants.py's docstring block and reproduced exactly
here, cross-checked against the source PDF's "Sampling" / "Edit distance
computation" / "Peakedness" subsections during pipeline construction (not
re-derived from memory):

    Peak(M;x) = (1/n) * sum_i I(ED(s_i, s_greedy) <= alpha * l_max)

`s_greedy` is the single temperature-0 (greedy) generation; `s_1..s_n` are
the `n` temperature-`CDD_SAMPLE_TEMPERATURE` samples (star topology: each
sample compared only against the greedy reference, never pairwise against
each other). `ED` is token-level edit (Levenshtein) distance; both sequences
are truncated to `CDD_MAX_TOKENS` (l_max=100) tokens before computing it.
`alpha=CDD_EDIT_DISTANCE_ALPHA=0.05` — the source PDF's own worked example
("With l=100 and α=0.05, a sample counts as close if it differs from the
greedy output by at most 5 token edits") confirms the threshold is
`alpha * l_max`, not `alpha * (actual truncated length)`, in the
all-sequences-truncated-to-l_max regime this pipeline always uses.
"""

from __future__ import annotations

from qcd.constants import CDD_EDIT_DISTANCE_ALPHA, CDD_MAX_TOKENS


def token_edit_distance(a: list[int], b: list[int]) -> int:
    """Levenshtein distance between two token-id sequences (unit cost
    insert/delete/substitute), classic O(len(a) * len(b)) DP."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,  # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution / match
            )
        prev, curr = curr, prev
    return prev[m]


def peakedness(
    greedy_token_ids: list[int],
    sample_token_ids_list: list[list[int]],
    *,
    max_tokens: int = CDD_MAX_TOKENS,
    alpha: float = CDD_EDIT_DISTANCE_ALPHA,
) -> float:
    """Peak(M;x) — fraction of `sample_token_ids_list` within the edit-
    distance threshold of `greedy_token_ids`, both truncated to
    `max_tokens`."""
    if not sample_token_ids_list:
        raise ValueError("peakedness needs at least one temperature sample")

    greedy_trunc = greedy_token_ids[:max_tokens]
    threshold = alpha * max_tokens

    close = 0
    for sample in sample_token_ids_list:
        sample_trunc = sample[:max_tokens]
        if token_edit_distance(greedy_trunc, sample_trunc) <= threshold:
            close += 1
    return close / len(sample_token_ids_list)
