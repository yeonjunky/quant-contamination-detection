"""Revision-pinned embedding adapter for TRACER semantic triage.

TRACER names ``jina-embeddings-v3`` but does not specify its model revision,
task adapter, similarity function, truncation, or pooling details.  This module
makes those reimplementation choices explicit and serializable.  Model loading
is lazy so importing the CPU-only decision pipeline never downloads weights.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_JINA_TASKS = {
    "retrieval.query",
    "retrieval.passage",
    "separation",
    "classification",
    "text-matching",
}


@dataclasses.dataclass(frozen=True)
class JinaEmbeddingConfig:
    """Auditable choices absent from the TRACER paper specification."""

    model_id: str = "jinaai/jina-embeddings-v3"
    revision: str = ""
    task: str = "text-matching"
    max_length: int = 8192
    truncate_dim: int | None = None
    similarity: str = "cosine_clipped_0_1"

    def __post_init__(self) -> None:
        if not _COMMIT_RE.fullmatch(self.revision):
            raise ValueError("revision must be a 40-character lowercase commit SHA")
        if self.task not in _JINA_TASKS:
            raise ValueError(f"unsupported jina task: {self.task}")
        if self.max_length < 1:
            raise ValueError("max_length must be positive")
        if self.truncate_dim is not None and self.truncate_dim < 1:
            raise ValueError("truncate_dim must be positive")
        if self.similarity != "cosine_clipped_0_1":
            raise ValueError("only cosine_clipped_0_1 similarity is supported")

    def manifest_record(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class JinaEmbeddingAdapter:
    """Lazy local model wrapper with deterministic score post-processing."""

    def __init__(
        self,
        config: JinaEmbeddingConfig,
        *,
        model_loader: Callable[[JinaEmbeddingConfig], Any] | None = None,
    ) -> None:
        self.config = config
        self._model_loader = model_loader or self._load_transformers_model
        self._model: Any | None = None

    @staticmethod
    def _load_transformers_model(config: JinaEmbeddingConfig) -> Any:
        from transformers import AutoModel

        # Remote model code is allowed only at the same immutable revision as
        # the weights.  This is both a reproducibility and supply-chain bound.
        return AutoModel.from_pretrained(
            config.model_id,
            revision=config.revision,
            code_revision=config.revision,
            trust_remote_code=True,
        )

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._model_loader(self.config)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.config.truncate_dim or 0), dtype=np.float32)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("all embedding inputs must be strings")
        arguments: dict[str, Any] = {
            "task": self.config.task,
            "max_length": self.config.max_length,
        }
        if self.config.truncate_dim is not None:
            arguments["truncate_dim"] = self.config.truncate_dim
        vectors = np.asarray(self.model.encode(list(texts), **arguments), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("embedding model returned an unexpected shape")
        if not np.isfinite(vectors).all():
            raise ValueError("embedding model returned a non-finite value")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("embedding model returned a zero vector")
        return vectors / norms

    def pairwise_scores(
        self,
        benchmark_descriptions: Sequence[str],
        training_descriptions: Sequence[str],
    ) -> np.ndarray:
        """Return the explicit cosine matrix used as TRACER's sigma."""
        benchmark = self.encode(benchmark_descriptions)
        training = self.encode(training_descriptions)
        if benchmark.shape[1] != training.shape[1]:
            raise ValueError("benchmark and training embedding dimensions differ")
        scores = benchmark @ training.T
        # The paper defines sigma on [0, 1] but does not disclose its exact
        # similarity transform.  We record this reimplementation choice:
        # normalized cosine, with negative values collapsed into the same
        # direct-U region.  It does not change routing at tau_low=0.6.
        return np.clip(scores, 0.0, 1.0)
