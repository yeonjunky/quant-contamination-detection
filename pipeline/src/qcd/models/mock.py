"""Deterministic synthetic model + tokenizer for the local dry-run harness.

No GPU, no model weights: MockModel/MockTokenizer expose the call surface
real loaders (models/loader.py, not yet written) must also expose, so the
rest of the pipeline never branches on mock-vs-real.

This is not random noise. The mock injects a *known* generative process so
the dry run can assert expected-signed pipeline outputs, not just "didn't
crash": the caller supplies whether a synthetic item is "contaminated" and a
quality level (ground truth only the test harness that created the item
would know), and the mock behaves the way a real memorized-vs-not model
plausibly would:
  - contaminated items produce peakier, more repeatable samples across
    temperature (low output diversity across the n=T=0.8 samples CDD needs)
    and higher-confidence (less negative) per-token log-probabilities
  - quality drives a fractional, non-0/1 partial test-pass rate

Everything is seeded from (item_id, sample_id, ...) via sha256, so re-running
the dry run reproduces identical numbers.
"""

import hashlib

import numpy as np
from transformers import GPT2TokenizerFast

_TOKENIZER: GPT2TokenizerFast | None = None


def _tokenizer() -> GPT2TokenizerFast:
    # Lazy singleton: a real, small, CPU-only tokenizer (vocab/merges files,
    # not model weights) so token ids and vocab size are realistic. Requires
    # network on first call to fetch from the HF Hub; cached under HF_HOME
    # thereafter.
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
    return _TOKENIZER


def _seed_from(*parts: object) -> int:
    """Deterministic 32-bit seed derived from arbitrary parts."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


class GenerationSample:
    def __init__(self, text: str, token_ids: list[int], token_logprobs: list[float], is_greedy: bool):
        self.text = text
        self.token_ids = token_ids
        self.token_logprobs = token_logprobs
        self.is_greedy = is_greedy


class MockTokenizer:
    """Thin wrapper around a real, small, CPU-only tokenizer (GPT-2's)."""

    def __init__(self) -> None:
        self._tok = _tokenizer()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size


class MockModel:
    """Synthetic stand-in for a real quantized code model.

    Callers supply the hidden ground truth (`contaminated`, `quality`) that a
    real model obviously isn't told. This exists purely to give the dry-run
    harness a known-answer generative process to check the rest of the
    pipeline against, not to simulate an actual model's behavior.
    """

    # Per-token log-prob confidence level, before per-call noise.
    _CONFIDENCE_CONTAMINATED = 0.92
    _CONFIDENCE_CLEAN = 0.55

    def __init__(self, tokenizer: MockTokenizer | None = None) -> None:
        self.tokenizer = tokenizer or MockTokenizer()

    def generate(
        self,
        item_id: str,
        prompt: str,
        *,
        contaminated: bool,
        temperature: float,
        sample_id: int,
    ) -> GenerationSample:
        is_greedy = temperature == 0.0
        vocab_size = self.tokenizer.vocab_size

        # Contaminated items reproduce the same token stream (length and
        # content) regardless of sample_id (a memorized answer doesn't move
        # with sampling temperature); clean, non-greedy samples get a fresh
        # draw per sample_id, giving the high sample-to-sample diversity CDD
        # expects from non-memorized items. Length must come from this same
        # seed, not a per-sample one, or "repeatable" would still vary by
        # sequence length even when the tokens themselves don't.
        stream_seed = _seed_from(item_id, 0 if (is_greedy or contaminated) else sample_id)
        stream_rng = np.random.RandomState(stream_seed)
        n_tokens = int(stream_rng.randint(8, 24))
        token_ids = stream_rng.randint(0, vocab_size, size=n_tokens).tolist()

        noise_rng = np.random.RandomState(_seed_from(item_id, sample_id, temperature, "noise"))
        confidence = self._confidence(contaminated, is_greedy, noise_rng)
        token_logprobs = self._score_tokens(token_ids, confidence, noise_rng)
        text = self.tokenizer.decode(token_ids)
        return GenerationSample(text=text, token_ids=token_ids, token_logprobs=token_logprobs, is_greedy=is_greedy)

    def score_logprobs(self, item_id: str, token_ids: list[int], *, contaminated: bool) -> list[float]:
        """Teacher-forced per-token log-probability for an already-generated
        sequence (needed by scoring/logprob.py, and by detectors like Min-k%
        that read the full per-token array rather than a summary scalar)."""
        rng = np.random.RandomState(_seed_from(item_id, "score"))
        confidence = self._confidence(contaminated, is_greedy=True, rng=rng)
        return self._score_tokens(token_ids, confidence, rng)

    def partial_pass_rate(self, item_id: str, quality: float) -> float:
        """Fractional (not 0/1) partial test-case pass rate, a deterministic
        function of quality plus small seeded noise — exercises the
        continuous scorer rather than a pass/fail boolean."""
        rng = np.random.RandomState(_seed_from(item_id, "pass_rate"))
        noise = rng.uniform(-0.05, 0.05)
        return float(np.clip(quality + noise, 0.0, 1.0))

    @classmethod
    def _confidence(cls, contaminated: bool, is_greedy: bool, rng: np.random.RandomState) -> float:
        base = cls._CONFIDENCE_CONTAMINATED if contaminated else cls._CONFIDENCE_CLEAN
        # Contaminated (and any greedy) samples stay near the base
        # confidence; clean, non-greedy samples jitter more, mirroring
        # genuine sampling variance from a non-memorized model.
        jitter = 0.0 if (is_greedy or contaminated) else rng.uniform(-0.15, 0.15)
        return float(np.clip(base + jitter, 0.05, 0.99))

    @staticmethod
    def _score_tokens(token_ids: list[int], confidence: float, rng: np.random.RandomState) -> list[float]:
        """Per-token log-prob of each chosen token: log(confidence) minus
        small seeded noise, always a valid (negative) log-probability."""
        noise = rng.uniform(0.0, 0.1, size=len(token_ids))
        probs = np.clip(confidence - noise, 1e-4, 0.999)
        return np.log(probs).tolist()
