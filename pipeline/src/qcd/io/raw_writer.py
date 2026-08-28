"""Item-level raw-data writer — the three-table parquet schema from
pipeline_build_plan.md, matching paper §5 step 8's requirement: "Store
item-level raw data for every condition... Aggregate-only storage would
foreclose the paired and mixed-effects analyses this design depends on."

- `items.parquet` — one row per item (id, dataset, difficulty, coarse legacy
  proxy, TRACER evidence, release/version pin).
- `model_item_labels.parquet` — one row per (model, item), carrying primary
  and sensitivity temporal labels plus the frozen boundaries that produced them.
- `generations.<part>.parquet` — one row per (model, quant, item, sample):
  generated text, full completion per-token logprob array, fixed prompt
  per-token logprob array on the greedy row, partial pass rate, decoding
  params, model/tokenizer revision hashes.
- `detector_scores.<part>.parquet` — one row per (model, quant, item, detector):
score, threshold used, source sample ids.

Rows are flushed in bounded, atomically replaced part files so an interrupted
run retains every completed batch without accumulating the entire experiment
in memory.

`Item.metadata` (a heterogeneous dict — LCB items and HumanEval+/MBPP+ items
carry different keys) is stored as a JSON string column rather than a
pyarrow struct column, since a struct column would need one consistent
schema across every row and this dict's shape varies by dataset.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import tempfile

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


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class RawDataWriter:
    def __init__(self, output_dir: str | Path, *, file_prefix: str = "") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_prefix = f"{file_prefix}_" if file_prefix else ""
        self._generation_rows: list[dict] = []
        self._detector_score_rows: list[dict] = []

    def write_items(self, items: list[Item]) -> Path:
        path = self.output_dir / f"{self.file_prefix}items.parquet"
        _write_parquet_atomic(pd.DataFrame([_item_to_row(item) for item in items]), path)
        return path

    def write_model_item_labels(self, rows: list[dict]) -> Path:
        path = self.output_dir / f"{self.file_prefix}model_item_labels.parquet"
        _write_parquet_atomic(pd.DataFrame(rows), path)
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
        passed: bool | None = None,
        decoding_temperature: float | None = None,
        generation_seconds: float | None = None,
        prompt_scoring_seconds: float | None = None,
        sandbox_scoring_seconds: float | None = None,
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
                "passed": passed,
                "decoding_temperature": decoding_temperature,
                "generation_seconds": generation_seconds,
                "prompt_scoring_seconds": prompt_scoring_seconds,
                "sandbox_scoring_seconds": sandbox_scoring_seconds,
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

    def flush(self, *, part: str | None = None) -> dict[str, Path]:
        """Atomically write and clear one bounded batch of buffered rows."""
        written = {}
        suffix = f".{part}" if part else ""
        if self._generation_rows:
            path = self.output_dir / f"{self.file_prefix}generations{suffix}.parquet"
            _write_parquet_atomic(pd.DataFrame(self._generation_rows), path)
            written["generations"] = path
            self._generation_rows.clear()
        if self._detector_score_rows:
            path = self.output_dir / f"{self.file_prefix}detector_scores{suffix}.parquet"
            _write_parquet_atomic(pd.DataFrame(self._detector_score_rows), path)
            written["detector_scores"] = path
            self._detector_score_rows.clear()
        return written

    @property
    def n_buffered_generations(self) -> int:
        return len(self._generation_rows)

    @property
    def n_buffered_detector_scores(self) -> int:
        return len(self._detector_score_rows)
