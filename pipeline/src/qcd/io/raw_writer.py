"""Item-level raw-data writer — the three-table parquet schema from
pipeline_build_plan.md, matching paper §5 step 8's requirement: "Store
item-level raw data for every condition... Aggregate-only storage would
foreclose the paired and mixed-effects analyses this design depends on."

- `items.parquet` — one row per item (id, dataset, difficulty, coarse legacy
  proxy, paper §4.2's per-model corpus-reference axis, release/version pin).
- `model_item_labels.parquet` — one row per (model, item), carrying primary
  and sensitivity temporal labels plus the frozen boundaries that produced them.
- `generations.<part>.parquet` — one row per (model, quant, item, sample):
  generated text, full completion per-token logprob array, fixed prompt
  per-token logprob array on the greedy row, partial pass rate, decoding
  params, model/tokenizer revision hashes, whether the generation stopped at
  the 512-token cap, and (on the greedy row) the scored-text identity, token
  boundaries and chat-template id paper §4.4 asks to be recorded with a
  probability-detector score.
- `chat_templates.parquet` — one row per (model, quant, chat template): the
  rendered template text itself, stored once per run rather than repeated on
  every generations row, keyed by the `chat_template_id` those rows carry.
- `detector_scores.<part>.parquet` — one row per (model, quant, item, detector):
score, threshold used, source sample ids.

New columns are added as optional keyword arguments defaulting to None, so a
parquet file written before they existed still reads — the columns are simply
absent from it.

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

from qcd.data.schema import Item, corpus_reference_to_json


def _item_to_row(item: Item) -> dict:
    return {
        "item_id": item.item_id,
        "dataset": item.dataset.value,
        "prompt": item.prompt,
        "difficulty": item.difficulty,
        "contamination_proxy": item.contamination_proxy,
        # Paper §4.2's separate, non-exclusive corpus-evidence axis, as a JSON
        # string for the same reason `metadata_json` is one: it holds a
        # variable-length list of (model, method family, stage) cells, not one
        # value per item. Replaces the single `tracer_label` float column an
        # older raw tree carries; `data.schema.corpus_reference_from_json`
        # reads both shapes.
        "corpus_reference_json": corpus_reference_to_json(item.corpus_reference),
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

    def write_chat_templates(self, rows: list[dict]) -> Path:
        """One row per (model, quant, chat template). Paper §4.4 requires the
        chat template to be recorded with the probability-detector scores; the
        template is identical for every item of a given tokenizer, so it is
        stored once per run and generations rows carry only its
        `chat_template_id`. Rewritten in full whenever a new template is seen,
        which is at most once per model/precision arm."""
        path = self.output_dir / f"{self.file_prefix}chat_templates.parquet"
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
        truncated_at_cap: bool | None = None,
        max_new_tokens: int | None = None,
        decoding_settings_id: str | None = None,
        chat_template_id: str | None = None,
        prompt_chat_template_applied: bool | None = None,
        prompt_target_text_sha256: str | None = None,
        prompt_target_char_span: tuple[int, int] | list[int] | None = None,
        prompt_target_token_indices: list[int] | None = None,
        prompt_rendered_char_length: int | None = None,
    ) -> None:
        """`truncated_at_cap` / `max_new_tokens` record paper §4.4's "we record
        per item whether generation stopped at the cap and report the
        truncated-generation rate by precision".

        The `prompt_*` fields and `chat_template_id` record §4.4's "Record
        target text, token boundaries, truncation, chat template,
        tokenizer/checkpoint revisions and decoding settings" for the
        probability detectors. They belong on the greedy row, next to
        `prompt_token_logprobs`, which is the only row that carries the
        fixed-text scoring pass. `decoding_settings_id` points at the run
        manifest's full resolved decoding-settings record.
        """
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
                "truncated_at_cap": truncated_at_cap,
                "max_new_tokens": max_new_tokens,
                "decoding_settings_id": decoding_settings_id,
                "chat_template_id": chat_template_id,
                "prompt_chat_template_applied": prompt_chat_template_applied,
                "prompt_target_text_sha256": prompt_target_text_sha256,
                "prompt_target_char_span": (
                    list(prompt_target_char_span)
                    if prompt_target_char_span is not None else None
                ),
                "prompt_target_token_indices": (
                    list(prompt_target_token_indices)
                    if prompt_target_token_indices is not None else None
                ),
                "prompt_rendered_char_length": prompt_rendered_char_length,
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
