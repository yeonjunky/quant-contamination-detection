"""Content-addressed generation cache, shared between the continuous-scoring
pipeline (paper §5 step 1) and the detector-scoring pipeline (step 2) so
CDD's multi-sample generation cost is paid once, not twice, per §4.4's
cost-sharing directive:

  "design steps 1 and 2 to share underlying generations wherever possible."

Local filesystem-backed (pickle payloads under a two-level hash-fanout
directory) — deliberately simple; swap the backend later if this becomes a
bottleneck, but nothing downstream should depend on the storage format.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pickle
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class CacheKey:
    model_name: str
    quant: str
    item_id: str
    is_greedy: bool
    sample_id: int
    prompt: str

    @property
    def digest(self) -> str:
        """Stable hash of every field, including the prompt text itself (not
        just its length) — if a prompt template changes, the cache misses
        rather than silently serving a stale generation for a different
        prompt under the same item_id."""
        parts = (
            self.model_name,
            self.quant,
            self.item_id,
            str(self.is_greedy),
            str(self.sample_id),
            self.prompt,
        )
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


class GenerationCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: CacheKey) -> Path:
        digest = key.digest
        # Two-level fanout so a full-scale run's cache directory doesn't put
        # hundreds of thousands of files in one flat directory.
        return self.cache_dir / digest[:2] / f"{digest}.pkl"

    def get(self, key: CacheKey):
        path = self._path_for(key)
        if not path.exists():
            return None
        with path.open("rb") as f:
            return pickle.load(f)

    def put(self, key: CacheKey, value) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(value, f)

    def __contains__(self, key: CacheKey) -> bool:
        return self._path_for(key).exists()
