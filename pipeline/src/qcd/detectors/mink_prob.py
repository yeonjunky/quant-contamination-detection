"""Min-k% Prob — Shi et al. 2023, *Detecting Pretraining Data from Large
Language Models* (arXiv:2310.16789, ICLR 2024; `pdfs/2310.16789.pdf`).
Formula pulled and verified directly from the source PDF during pipeline
construction (equation 1, "Peakedness"-adjacent "MIN-K% PROB" section), not
guessed:

    Min-K%(x) = the k% of tokens in x with the minimum token probability
    MIN-K%PROB(x) = (1/E) * sum_{x_i in Min-K%(x)} log p(x_i | x_1..x_{i-1})

where E = |Min-K%(x)|. "If the average log likelihood is high, the text is
likely in the pretraining data" (source PDF, §1) — i.e. this score, like
perplexity's `negative_log_perplexity_score`, is oriented "higher = more
contaminated-looking" already; no sign flip needed.

`k=20` is the paper's own reported best value from its k in {10,20,30,40,50}
validation sweep — used as this module's default, overridable per call.
"""

from __future__ import annotations

DEFAULT_K_PERCENT = 20.0


def mink_prob(token_logprobs: list[float], *, k_percent: float = DEFAULT_K_PERCENT) -> float:
    if not token_logprobs:
        raise ValueError("mink_prob needs at least one token log-probability")
    if not 0 < k_percent <= 100:
        raise ValueError(f"k_percent must be in (0, 100], got {k_percent}")

    n_selected = max(1, round(len(token_logprobs) * k_percent / 100))
    lowest = sorted(token_logprobs)[:n_selected]  # log-probabilities: most negative = lowest probability
    return sum(lowest) / len(lowest)
