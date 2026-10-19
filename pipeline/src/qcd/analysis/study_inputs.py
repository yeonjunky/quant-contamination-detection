"""Read a main-study raw tree into the exact arrays §4.5.5, §4.5.6 and §4.4
ask for.

This module is the only place that knows the on-disk column names. They come
from `io/raw_writer.py` and `data/schema.py`, not from a guess:

- `items.parquet` — `item_id`, `dataset` (`data.schema.Dataset` values, so
  LiveCodeBench is `lcb_pre` / `lcb_post`), `release_version`.
- `model_item_labels.parquet` — `model`, `item_id`, `dataset`,
  `primary_label` / `sensitivity_label` (`data.schema.TemporalProxyLabel`
  values: `possible-exposure`, `clean-by-model-cutoff`,
  `shared-clean-control`), `boundary_ambiguous`.
- `detector_scores.<part>.parquet` — `model`, `quant`, `item_id`, `detector`,
  `score`. `real_run.py` writes the detector names `cdd`, `perplexity`,
  `mink_prob` (plus the `completion_*` diagnostics, which §4.4 keeps
  separate and which no confirmatory test uses).
- `generations.<part>.parquet` — `model`, `quant`, `item_id`, `sample_id`,
  `is_greedy`, `passed`, `truncated_at_cap`, `max_new_tokens`.

Three things this module deliberately does *not* do: flip a score's sign
(`qcd.detectors` already stores every score so that larger = more possible
exposure), choose which items enter a test on the basis of their scores, or
silently drop an incomplete cell. Every selection returns its own counts so
the analysis output can report what was dropped and why.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from qcd.data.schema import Dataset, TemporalProxyLabel
from qcd.detectors.cdd import peakedness_fixed_threshold

#: The two `data.schema.Dataset` values that are LiveCodeBench. §4.5.6's
#: C1-C3 run on "all 1,055 LCB release_v6 items", which is both of them.
LCB_DATASETS: tuple[str, ...] = (Dataset.LCB_PRE.value, Dataset.LCB_POST.value)

#: §4.5.6's C4 item sets, in `TemporalProxyLabel` spelling.
LABEL_POSSIBLE_EXPOSURE = TemporalProxyLabel.POSSIBLE_EXPOSURE.value
LABEL_SHARED_CLEAN_CONTROL = TemporalProxyLabel.SHARED_CLEAN_CONTROL.value

#: The three §4.4 detectors the confirmatory family uses, spelled as
#: `real_run.py` writes them.
CONFIRMATORY_DETECTORS: tuple[str, ...] = ("perplexity", "mink_prob", "cdd")

_ITEM_COLUMNS = ("item_id", "dataset")
_LABEL_COLUMNS = ("model", "item_id", "primary_label")
_DETECTOR_COLUMNS = ("model", "quant", "item_id", "detector", "score")
_GENERATION_COLUMNS = ("model", "quant", "item_id", "sample_id", "is_greedy")


@dataclasses.dataclass(frozen=True)
class RawTables:
    """The four tables of one run directory, already de-duplicated."""

    raw_dir: Path
    items: pd.DataFrame
    model_item_labels: pd.DataFrame
    detector_scores: pd.DataFrame
    generations: pd.DataFrame
    source_files: dict[str, list[str]]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], path: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s) {missing}; expected the "
            "io/raw_writer.py schema"
        )


def _read_parts(raw_dir: Path, stem: str, *, file_prefix: str = "") -> tuple[pd.DataFrame, list[str]]:
    """Read `<stem>.parquet` plus every `<stem>.<part>.parquet` written by
    `RawDataWriter.flush(part=...)`."""
    paths = sorted(raw_dir.glob(f"{file_prefix}{stem}.parquet")) + sorted(
        raw_dir.glob(f"{file_prefix}{stem}.*.parquet")
    )
    if not paths:
        raise FileNotFoundError(
            f"no {file_prefix}{stem}*.parquet under {raw_dir} — the analysis needs the "
            "item-level raw tables written by io/raw_writer.py"
        )
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True), [path.name for path in paths]


def _dedupe(frame: pd.DataFrame, keys: list[str], values: list[str], what: str) -> pd.DataFrame:
    """Drop rows a resumed run wrote twice, but refuse to choose between two
    *different* values for the same key — that is a corrupt tree, not a
    duplicate. `values` names the columns whose disagreement makes two rows
    genuinely different measurements."""
    duplicated = frame.duplicated(subset=keys, keep=False)
    if not duplicated.any():
        return frame
    repeated = frame[duplicated]
    for value in values:
        if value not in frame.columns:
            continue
        conflicting = repeated.groupby(keys, dropna=False)[value].nunique(dropna=False) > 1
        if bool(conflicting.any()):
            offenders = conflicting[conflicting].index[:5].tolist()
            raise ValueError(
                f"{what}: the same {keys} appears with different {value!r} values "
                f"(first offenders: {offenders}). Re-write the raw tree rather than "
                "letting the analysis pick one."
            )
    return frame.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)


def load_raw_tables(run_dir: str | Path, *, file_prefix: str = "") -> RawTables:
    """Load one run's four raw tables.

    `run_dir` may be the run directory (`real_run.py` writes the parquet
    files into its `raw/` subdirectory, next to `manifest.json`) or that
    `raw/` directory itself.
    """
    run_dir = Path(run_dir)
    raw_dir = run_dir / "raw" if (run_dir / "raw").is_dir() else run_dir

    items = pd.read_parquet(raw_dir / f"{file_prefix}items.parquet")
    _require_columns(items, _ITEM_COLUMNS, str(raw_dir / "items.parquet"))
    labels = pd.read_parquet(raw_dir / f"{file_prefix}model_item_labels.parquet")
    _require_columns(labels, _LABEL_COLUMNS, str(raw_dir / "model_item_labels.parquet"))
    scores, score_files = _read_parts(raw_dir, "detector_scores", file_prefix=file_prefix)
    _require_columns(scores, _DETECTOR_COLUMNS, "detector_scores*.parquet")
    generations, generation_files = _read_parts(raw_dir, "generations", file_prefix=file_prefix)
    _require_columns(generations, _GENERATION_COLUMNS, "generations*.parquet")

    scores = _dedupe(
        scores, ["model", "quant", "item_id", "detector"], ["score"], "detector_scores"
    )
    generations = _dedupe(
        generations,
        ["model", "quant", "item_id", "sample_id"],
        ["is_greedy", "passed", "truncated_at_cap"],
        "generations",
    )
    return RawTables(
        raw_dir=raw_dir,
        items=items,
        model_item_labels=labels,
        detector_scores=scores,
        generations=generations,
        source_files={
            "items": [f"{file_prefix}items.parquet"],
            "model_item_labels": [f"{file_prefix}model_item_labels.parquet"],
            "detector_scores": score_files,
            "generations": generation_files,
        },
    )


def lcb_item_ids(tables: RawTables) -> pd.Index:
    """Every LiveCodeBench item id in the run, sorted — §4.5.6's C1-C3 sample
    before any completeness filtering."""
    lcb = tables.items[tables.items["dataset"].isin(LCB_DATASETS)]
    return pd.Index(sorted(lcb["item_id"].astype(str).unique()))


def _score_matrix(
    tables: RawTables,
    *,
    model: str,
    quants: tuple[str, ...],
    detectors: tuple[str, ...],
    item_ids: pd.Index,
) -> pd.DataFrame:
    """`(item_id) x (detector, quant)` score frame, restricted to `item_ids`."""
    scores = tables.detector_scores
    selected = scores[
        (scores["model"] == model)
        & (scores["quant"].isin(quants))
        & (scores["detector"].isin(detectors))
        & (scores["item_id"].astype(str).isin(set(item_ids)))
    ].copy()
    selected["item_id"] = selected["item_id"].astype(str)
    wide = selected.pivot(index="item_id", columns=["detector", "quant"], values="score")
    return wide.reindex(item_ids)


def _complete_rows(wide: pd.DataFrame, required: list[tuple[str, str]]) -> pd.Series:
    """Rows whose every required (detector, precision) score is present and
    finite. A missing pivot cell reads as NaN, which is not finite."""
    values = np.asarray(wide[required].to_numpy(), dtype=float)
    return pd.Series(np.isfinite(values).all(axis=1), index=wide.index)


@dataclasses.dataclass(frozen=True)
class PairedDetectorScores:
    """C1-C3's input: aligned ``(baseline, target)`` arrays per detector, plus
    the counts behind them."""

    arrays: dict[str, tuple[np.ndarray, np.ndarray]]
    item_ids: list[str]
    n_lcb_items: int
    n_complete_items: int
    n_items_dropped_incomplete: int
    detectors_missing: list[str]


def paired_detector_scores(
    tables: RawTables,
    *,
    model: str,
    baseline: str,
    target: str,
    detectors: tuple[str, ...] = CONFIRMATORY_DETECTORS,
) -> PairedDetectorScores:
    """§4.5.6's C1-C3 sample: every LCB item of `model`, scored at both
    precisions by all three detectors. No exposure label is used.

    An item missing any of the six required scores cannot enter a *paired*
    test; it is dropped from all three arrays together (so the three tests
    stay on one common item set) and counted in
    `n_items_dropped_incomplete`.
    """
    item_ids = lcb_item_ids(tables)
    wide = _score_matrix(
        tables,
        model=model,
        quants=(baseline, target),
        detectors=detectors,
        item_ids=item_ids,
    )
    required = [(detector, quant) for detector in detectors for quant in (baseline, target)]
    missing_columns = [column for column in required if column not in wide.columns]
    detectors_missing = sorted({detector for detector, _ in missing_columns})
    if detectors_missing:
        return PairedDetectorScores(
            arrays={},
            item_ids=[],
            n_lcb_items=int(item_ids.size),
            n_complete_items=0,
            n_items_dropped_incomplete=int(item_ids.size),
            detectors_missing=detectors_missing,
        )

    complete = _complete_rows(wide, required)
    kept = wide[complete]
    arrays = {
        detector: (
            kept[(detector, baseline)].to_numpy(dtype=float),
            kept[(detector, target)].to_numpy(dtype=float),
        )
        for detector in detectors
    }
    return PairedDetectorScores(
        arrays=arrays,
        item_ids=[str(value) for value in kept.index],
        n_lcb_items=int(item_ids.size),
        n_complete_items=int(kept.shape[0]),
        n_items_dropped_incomplete=int(item_ids.size - kept.shape[0]),
        detectors_missing=[],
    )


@dataclasses.dataclass(frozen=True)
class C4Inputs:
    """C4's input: the six score arrays plus the label vector, all aligned to
    one item order."""

    baseline_scores: dict[str, np.ndarray]
    target_scores: dict[str, np.ndarray]
    labels: np.ndarray
    item_ids: list[str]
    n_possible_exposure: int
    n_shared_clean_control: int
    n_items_dropped_incomplete: int
    label_field: str


def c4_label_frame(
    tables: RawTables, *, model: str, label_field: str = "primary_label"
) -> pd.DataFrame:
    """The `possible-exposure` / `shared-clean-control` LCB items of one model
    (§4.5.6 uses exactly these two groups; `clean-by-model-cutoff` items are
    outside C4's item set)."""
    labels = tables.model_item_labels
    if label_field not in labels.columns:
        raise ValueError(f"model_item_labels.parquet has no column {label_field!r}")
    lcb = set(lcb_item_ids(tables))
    selected = labels[
        (labels["model"] == model)
        & (labels[label_field].isin([LABEL_POSSIBLE_EXPOSURE, LABEL_SHARED_CLEAN_CONTROL]))
        & (labels["item_id"].astype(str).isin(lcb))
    ].copy()
    selected["item_id"] = selected["item_id"].astype(str)
    return selected[["item_id", label_field]].drop_duplicates().sort_values("item_id")


def c4_inputs(
    tables: RawTables,
    *,
    model: str,
    baseline: str,
    target: str,
    detectors: tuple[str, ...] = CONFIRMATORY_DETECTORS,
    label_field: str = "primary_label",
) -> C4Inputs:
    """§4.5.6's C4 input, on the labelled LCB subset only."""
    label_frame = c4_label_frame(tables, model=model, label_field=label_field)
    item_ids = pd.Index(label_frame["item_id"].tolist())
    wide = _score_matrix(
        tables,
        model=model,
        quants=(baseline, target),
        detectors=detectors,
        item_ids=item_ids,
    )
    required = [(detector, quant) for detector in detectors for quant in (baseline, target)]
    if any(column not in wide.columns for column in required):
        return C4Inputs(
            baseline_scores={},
            target_scores={},
            labels=np.zeros(0, dtype=bool),
            item_ids=[],
            n_possible_exposure=0,
            n_shared_clean_control=0,
            n_items_dropped_incomplete=int(item_ids.size),
            label_field=label_field,
        )
    complete = _complete_rows(wide, required)
    kept_ids = [str(value) for value in wide.index[complete]]
    kept = wide[complete]
    label_lookup = dict(zip(label_frame["item_id"], label_frame[label_field]))
    labels = np.array(
        [label_lookup[item_id] == LABEL_POSSIBLE_EXPOSURE for item_id in kept_ids], dtype=bool
    )
    return C4Inputs(
        baseline_scores={
            detector: kept[(detector, baseline)].to_numpy(dtype=float) for detector in detectors
        },
        target_scores={
            detector: kept[(detector, target)].to_numpy(dtype=float) for detector in detectors
        },
        labels=labels,
        item_ids=kept_ids,
        n_possible_exposure=int(labels.sum()),
        n_shared_clean_control=int((~labels).sum()),
        n_items_dropped_incomplete=int(item_ids.size - len(kept_ids)),
        label_field=label_field,
    )


def outcome_frame(
    tables: RawTables, *, label_field: str = "primary_label"
) -> pd.DataFrame:
    """§4.5.5's tidy input: one row per (model, item, precision) with
    `correct` and `exposure_proxy` under §3.1's coding.

    `correct` is the greedy row's `passed` (§4.4: "pass@1 is scored from the
    single greedy output ... not estimated from the 50 temperature samples").
    `exposure_proxy` is True for `possible-exposure` and False for
    `shared-clean-control`; every other label is outside §3.1's 2x2 and is
    dropped. Rows whose `passed` was never written are dropped.
    """
    generations = tables.generations
    if "passed" not in generations.columns:
        raise ValueError(
            "generations*.parquet has no `passed` column — §4.5.5's outcome is pass@1 "
            "from the greedy row"
        )
    greedy = generations[generations["is_greedy"].astype(bool)].copy()
    greedy["item_id"] = greedy["item_id"].astype(str)
    greedy = greedy[greedy["passed"].notna()]

    labels = tables.model_item_labels.copy()
    labels["item_id"] = labels["item_id"].astype(str)
    labels = labels[
        labels[label_field].isin([LABEL_POSSIBLE_EXPOSURE, LABEL_SHARED_CLEAN_CONTROL])
    ][["model", "item_id", label_field]].drop_duplicates()

    lcb = set(lcb_item_ids(tables))
    merged = greedy.merge(labels, on=["model", "item_id"], how="inner")
    merged = merged[merged["item_id"].isin(lcb)]
    return pd.DataFrame(
        {
            "model": merged["model"].astype(str),
            "item_id": merged["item_id"].astype(str),
            "precision": merged["quant"].astype(str),
            "exposure_proxy": merged[label_field] == LABEL_POSSIBLE_EXPOSURE,
            "correct": merged["passed"].astype(bool).astype(int),
        }
    ).reset_index(drop=True)


def truncated_generation_rates(tables: RawTables) -> dict:
    """§4.4's "report the truncated-generation rate by precision".

    Two rates per cell, because they answer different questions: the greedy
    rate is the one §4.4 ties to pass@1 ("a truncation rate that differs
    across precisions would confound a pass@1 shift with a length-cap
    artifact"), and the all-generations rate covers the 50 CDD samples too.
    Generations whose `truncated_at_cap` flag was never written are counted
    separately and excluded from both denominators rather than assumed False.
    """
    generations = tables.generations
    if "truncated_at_cap" not in generations.columns:
        raise ValueError(
            "generations*.parquet has no `truncated_at_cap` column — §4.4 requires the "
            "per-item cap flag"
        )

    def summarize(frame: pd.DataFrame) -> dict:
        flag = frame["truncated_at_cap"]
        known = flag.notna()
        n_known = int(known.sum())
        n_truncated = int(flag[known].astype(bool).sum())
        return {
            "n_generations": int(frame.shape[0]),
            "n_flag_missing": int(frame.shape[0] - n_known),
            "n_truncated": n_truncated,
            "truncated_rate": (n_truncated / n_known) if n_known else None,
        }

    caps = (
        sorted({int(value) for value in generations["max_new_tokens"].dropna().unique()})
        if "max_new_tokens" in generations.columns
        else []
    )
    greedy_mask = generations["is_greedy"].astype(bool)

    by_model_precision = []
    for (model, quant), cell in generations.groupby(["model", "quant"], dropna=False):
        by_model_precision.append(
            {
                "model": str(model),
                "precision": str(quant),
                "all_generations": summarize(cell),
                "greedy_only": summarize(cell[cell["is_greedy"].astype(bool)]),
            }
        )

    by_precision = []
    for quant, cell in generations.groupby("quant", dropna=False):
        by_precision.append(
            {
                "precision": str(quant),
                "all_generations": summarize(cell),
                "greedy_only": summarize(cell[cell["is_greedy"].astype(bool)]),
            }
        )

    return {
        "definition": (
            "§4.4: the 512-token generation cap is shorter than LiveCodeBench's official "
            "runner default, so each generation records whether it stopped at the cap; the "
            "rate is reported by precision. `greedy_only` is the pass@1-relevant rate."
        ),
        "max_new_tokens_observed": caps,
        "n_greedy_generations": int(greedy_mask.sum()),
        "by_precision": sorted(by_precision, key=lambda row: row["precision"]),
        "by_model_and_precision": sorted(
            by_model_precision, key=lambda row: (row["model"], row["precision"])
        ),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def cdd_length_and_empty_outputs(tables: RawTables) -> dict:
    """Descriptive, per model and precision: CDD's threshold length `l`, how
    often outputs are empty, and how often they hit the generation cap.

    `l` is the longest output after truncation to 100 tokens, so the CDD
    threshold moves with output length; an item whose outputs are all empty
    has l=0 and scores 1.0. These are reported next to C1-C3 so a reader can
    see whether a CDD shift travels with a length or empty-output shift.
    Nothing here enters C1-C4 or any other test. A `cdd` row written without
    the length fields is counted in `n_items_not_recorded`."""
    scores = tables.detector_scores
    cdd = scores[scores["detector"] == "cdd"]
    generations = tables.generations
    greedy_mask = generations["is_greedy"].astype(bool)
    rows = []
    for (model, quant), cell in cdd.groupby(["model", "quant"]):
        recorded = cell[cell["cdd_threshold_length"].notna()]
        lengths = recorded["cdd_threshold_length"].astype(int)
        n_samples = int(recorded["source_sample_ids"].map(len).sum()) - len(recorded)
        in_cell = (generations["model"] == model) & (generations["quant"] == quant)
        greedy_cap = generations.loc[in_cell & greedy_mask, "truncated_at_cap"].dropna().astype(bool)
        sample_cap = generations.loc[in_cell & ~greedy_mask, "truncated_at_cap"].dropna().astype(bool)
        rows.append({
            "model": str(model),
            "precision": str(quant),
            "n_items": int(len(cell)),
            "n_items_not_recorded": int(len(cell) - len(recorded)),
            "median_threshold_length": float(lengths.median()) if len(lengths) else None,
            "n_items_threshold_length_zero": int((lengths == 0).sum()),
            "greedy_empty_rate": _rate(
                int(recorded["cdd_greedy_empty"].astype(bool).sum()), len(recorded)
            ),
            "sample_empty_rate": _rate(int(recorded["cdd_n_empty_samples"].sum()), n_samples),
            "greedy_cap_hit_rate": _rate(int(greedy_cap.sum()), len(greedy_cap)),
            "sample_cap_hit_rate": _rate(int(sample_cap.sum()), len(sample_cap)),
        })
    return {
        "status": "descriptive and exploratory; not part of C1-C4 or any confirmatory computation",
        "definitions": {
            "threshold_length": (
                "CDD's l: the longest of the greedy output and the samples after truncation "
                "to 100 tokens (detectors.cdd.threshold_length)"
            ),
            "empty": "a generation with zero tokens",
            "cap_hit": (
                "a generation that stopped at the max_new_tokens cap; flags never written are "
                "left out of the denominator"
            ),
        },
        "by_model_and_precision": sorted(rows, key=lambda row: (row["model"], row["precision"])),
    }


def cdd_recomputed_scores(tables: RawTables) -> pd.DataFrame:
    """Per model, precision and LiveCodeBench item, recomputed from the stored
    token ids: `fixed_threshold_score` (CDD with the threshold held at
    alpha * 100 tokens, `detectors.cdd.peakedness_fixed_threshold`) and
    `all_outputs_empty` (greedy and every sample empty, so the recorded score
    is 1.0 by construction). An item without a greedy row or without samples
    is left out. Exploratory inputs only; C1-C4 never read them."""
    lcb = set(lcb_item_ids(tables))
    generations = tables.generations
    selected = generations[generations["item_id"].astype(str).isin(lcb)]
    rows = []
    for (model, quant, item_id), item in selected.groupby(["model", "quant", "item_id"]):
        greedy = item[item["is_greedy"].astype(bool)]
        samples = item[~item["is_greedy"].astype(bool)].sort_values("sample_id")
        if len(greedy) != 1 or samples.empty:
            continue
        greedy_ids = list(greedy["token_ids"].iloc[0])
        sample_ids = [list(ids) for ids in samples["token_ids"]]
        rows.append({
            "model": str(model),
            "quant": str(quant),
            "item_id": str(item_id),
            "fixed_threshold_score": peakedness_fixed_threshold(greedy_ids, sample_ids),
            "all_outputs_empty": not greedy_ids and not any(sample_ids),
        })
    return pd.DataFrame(
        rows,
        columns=["model", "quant", "item_id", "fixed_threshold_score", "all_outputs_empty"],
    )


@dataclasses.dataclass(frozen=True)
class CddSensitivityPairs:
    """Aligned ``(baseline, target)`` CDD arrays for the three exploratory
    variants, on the LCB items where the recorded score and the recomputed
    values exist at both precisions."""

    recorded: tuple[np.ndarray, np.ndarray]
    fixed_threshold: tuple[np.ndarray, np.ndarray]
    recorded_excluding_empty: tuple[np.ndarray, np.ndarray]
    n_lcb_items: int
    n_paired_items: int
    n_items_all_outputs_empty: int


def cdd_sensitivity_pairs(
    tables: RawTables,
    recomputed: pd.DataFrame,
    *,
    model: str,
    baseline: str,
    target: str,
) -> CddSensitivityPairs:
    """`recorded` is the CDD score as written (C3's definition); the excluding-
    empty variant drops an item whose outputs are all empty at either
    precision, since its score is 1.0 whatever the model did."""
    item_ids = lcb_item_ids(tables)
    wide = _score_matrix(
        tables, model=model, quants=(baseline, target), detectors=("cdd",), item_ids=item_ids
    )
    cell = recomputed[recomputed["model"] == model]
    fixed = cell.pivot(index="item_id", columns="quant", values="fixed_threshold_score")
    empty = cell.pivot(index="item_id", columns="quant", values="all_outputs_empty")
    fixed = fixed.reindex(index=item_ids, columns=[baseline, target])
    empty = empty.reindex(index=item_ids, columns=[baseline, target])

    recorded = pd.DataFrame(
        {quant: wide[("cdd", quant)] if ("cdd", quant) in wide.columns else np.nan
         for quant in (baseline, target)},
        index=item_ids,
    )
    complete = (
        np.isfinite(recorded.to_numpy(dtype=float)).all(axis=1)
        & np.isfinite(fixed.to_numpy(dtype=float)).all(axis=1)
    )
    any_empty = empty.fillna(False).astype(bool).any(axis=1).to_numpy()
    keep_nonempty = complete & ~any_empty

    def arrays(frame: pd.DataFrame, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kept = frame[mask]
        return kept[baseline].to_numpy(dtype=float), kept[target].to_numpy(dtype=float)

    return CddSensitivityPairs(
        recorded=arrays(recorded, complete),
        fixed_threshold=arrays(fixed, complete),
        recorded_excluding_empty=arrays(recorded, keep_nonempty),
        n_lcb_items=int(item_ids.size),
        n_paired_items=int(complete.sum()),
        n_items_all_outputs_empty=int((complete & any_empty).sum()),
    )
