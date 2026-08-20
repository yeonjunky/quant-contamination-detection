"""Teacher-forced per-token log-probability scoring entry points.

Generated-completion scoring remains available as confidence raw data. The
paper's probability-based contamination detectors use the independent fixed-
prompt entry point so every precision scores identical benchmark text.
"""

from __future__ import annotations


def score_item_logprobs(model, item_id: str, token_ids: list[int]) -> list[float]:
    return model.score_logprobs(item_id, token_ids)


def score_prompt_logprobs(model, item_id: str, prompt: str) -> list[float]:
    """Return log-probabilities for fixed benchmark-prompt tokens only."""
    return model.score_prompt_logprobs(item_id, prompt)
