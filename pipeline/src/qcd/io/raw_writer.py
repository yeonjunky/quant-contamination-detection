"""Item-level raw-data writer — the three-table parquet schema from
pipeline_build_plan.md, matching paper §5 step 8's requirement: "Store
item-level raw data for every condition... Aggregate-only storage would
foreclose the paired and mixed-effects analyses this design depends on."

- `items.parquet` — one row per item (id, dataset, condition, difficulty,
  contamination proxy label, TRACER label, release/version pin).
- `generations.parquet` — one row per (model, quant, item, sample):
  generated text, full completion per-token logprob array, fixed prompt
  per-token logprob array on the greedy row, partial pass rate, decoding
  params, model/tokenizer revision hashes.
- `detector_scores.parquet` — one row per (model, quant, item, detector):
  score, threshold used, source sample ids.

`Item.metadata` (a heterogeneous dict — LCB items and HumanEval+/MBPP+ items
carry different keys) is stored as a JSON string column rather than a
pyarrow struct column, since a struct column would need one consistent
schema across every row and this dict's shape varies by dataset.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd

from qcd.data.schema import Item


def _item_to_row(item: Item) -> dict:
    return {
        "item_id": item.item_id,
        "dataset": item.dataset.value,
        "prompt": item.prompt,
        "difficulty": item.difficulty,
        "contamination_proxy": item.contamination_proxy,
        "tracer_label": item.tracer_label,
        "release_version": item.release_version,
        "metadata_json": json.dumps(item.metadata, default=str),
    }


class RawDataWriter:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._generation_rows: list[dict] = []
        self._detector_score_rows: list[dict] = []

    def write_items(self, items: list[Item]) -> Path:
        path = self.output_dir / "items.parquet"
        pd.DataFrame([_item_to_row(item) for item in items]).to_parquet(path, index=False)
        return path

    def add_generation(
        self,
        *,
        model: str,
        quant: str,
        item_id: str,
        sample_id: int,
        is_greedy: bool,
        text: str,
        token_ids: list[int],
        token_logprobs: list[float],
        prompt_token_logprobs: list[float] | None = None,
        partial_pass_rate: float | None = None,
        decoding_temperature: float | None = None,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
    ) -> None:
        self._generation_rows.append(
            {
                "model": model,
                "quant": quant,
                "item_id": item_id,
                "sample_id": sample_id,
                "is_greedy": is_greedy,
                "text": text,
                "token_ids": list(token_ids),
                "token_logprobs": list(token_logprobs),
                "prompt_token_logprobs": (
                    list(prompt_token_logprobs)
                    if prompt_token_logprobs is not None else None
                ),
                "partial_pass_rate": partial_pass_rate,
                "decoding_temperature": decoding_temperature,
                "model_revision": model_revision,
                "tokenizer_revision": tokenizer_revision,
            }
        )

    def add_detector_score(
        self,
        *,
        model: str,
        quant: str,
        item_id: str,
        detector: str,
        score: float,
        threshold_used: float | None = None,
        source_sample_ids: list[int] | None = None,
    ) -> None:
        self._detector_score_rows.append(
            {
                "model": model,
                "quant": quant,
                "item_id": item_id,
                "detector": detector,
                "score": score,
                "threshold_used": threshold_used,
                "source_sample_ids": list(source_sample_ids) if source_sample_ids is not None else None,
            }
        )

    def flush(self) -> dict[str, Path]:
        """Writes buffered generation/detector-score rows to their parquet
        files (full overwrite of each file's current buffer contents — this
        is a batch writer, not an incremental appender; fine for pilot-scale
        runs, a known scaling concern flagged here for the full main run)."""
        written = {}
        if self._generation_rows:
            path = self.output_dir / "generations.parquet"
            pd.DataFrame(self._generation_rows).to_parquet(path, index=False)
            written["generations"] = path
        if self._detector_score_rows:
            path = self.output_dir / "detector_scores.parquet"
            pd.DataFrame(self._detector_score_rows).to_parquet(path, index=False)
            written["detector_scores"] = path
        return written

    @property
    def n_buffered_generations(self) -> int:
        return len(self._generation_rows)

    @property
    def n_buffered_detector_scores(self) -> int:
        return len(self._detector_score_rows)
