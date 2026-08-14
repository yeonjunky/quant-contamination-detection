"""Teacher-forced per-token log-probability scoring — a thin wrapper around
`LoadedModel.score_logprobs()` (models/loader.py's Protocol). Kept as its
own module (rather than inlined at every call site) because detectors that
need the *full* per-token array (perplexity, and especially Min-k% Prob,
which needs the actual lowest-k% subset, not a summary scalar) should read
through one place, not re-derive token ids from generated text themselves.
"""

from __future__ import annotations


def score_item_logprobs(model, item_id: str, token_ids: list[int]) -> list[float]:
    return model.score_logprobs(item_id, token_ids)
