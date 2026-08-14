"""Perplexity — one of the two probability-based detectors §2.4 identifies
as the primary detector family (the other is Min-k% Prob, detectors/
mink_prob.py). Standard definition: exponentiated mean negative per-token
log-probability over the (teacher-forced) sequence.

Lower perplexity <-> higher model confidence <-> the item is more plausibly
memorized/contaminated — the same direction Min-k% Prob and CDD's
peakedness use, so all three detector scores in this package are oriented
"higher score = more contaminated-looking" for AUC computation
(analysis/auc.py's `empirical_auc` treats `labels=True` as the positive/
contaminated class).
"""

from __future__ import annotations

import math


def perplexity(token_logprobs: list[float]) -> float:
    if not token_logprobs:
        raise ValueError("perplexity needs at least one token log-probability")
    mean_logprob = sum(token_logprobs) / len(token_logprobs)
    return math.exp(-mean_logprob)


def negative_log_perplexity_score(token_logprobs: list[float]) -> float:
    """`-log(perplexity)` = mean log-probability itself — monotonically
    equivalent to perplexity for ranking/AUC purposes, but oriented so
    *higher* means *more contaminated-looking* (perplexity itself runs the
    other way: lower perplexity = more confident = more contaminated-
    looking). Use this one as the actual detector score fed into AUC/paired
    comparisons; keep `perplexity()` around for the human-readable statistic."""
    if not token_logprobs:
        raise ValueError("negative_log_perplexity_score needs at least one token log-probability")
    return sum(token_logprobs) / len(token_logprobs)
