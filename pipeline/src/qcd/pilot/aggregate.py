"""Build the paper's pilot summaries from item-level parquet raw data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qcd.analysis.auc import empirical_auc, items_needed_for_delta_auc
from qcd.analysis.mixed_effects import fit_precision_contamination_glmm
from qcd.analysis.power import items_needed_diff_in_diff, items_needed_paired_t
from qcd.pilot.cdd_gate import check_cdd_gate
from qcd.pilot.pilot_report import base_rate, cohens_d_paired, pearson_r

PRIMARY_DETECTORS = ("cdd", "perplexity", "mink_prob")


def _read_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = run_dir / "raw"
    paths = [raw_dir / name for name in (
        "items.parquet", "generations.parquet", "detector_scores.parquet"
    )]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing pilot raw table(s): {', '.join(missing)}")
    return tuple(pd.read_parquet(path) for path in paths)  # type: ignore[return-value]


def _paired_scores(
    scores: pd.DataFrame, model: str, detector: str, baseline: str, target: str
) -> pd.DataFrame:
    subset = scores[(scores["model"] == model) & (scores["detector"] == detector)]
    paired = subset.pivot(index="item_id", columns="quant", values="score")
    if baseline not in paired or target not in paired:
        raise ValueError(f"missing {baseline}/{target} scores for {model}/{detector}")
    return paired[[baseline, target]].dropna()


def _did_percentage_points(generations: pd.DataFrame, items: pd.DataFrame, baseline: str, target: str) -> float:
    data = generations.merge(items[["item_id", "contamination_proxy"]], on="item_id")
    rates = data.groupby(["quant", "contamination_proxy"])["passed"].mean()
    required = [(q, label) for q in (baseline, target) for label in (False, True)]
    if any(key not in rates.index for key in required):
        raise ValueError("Q2 needs contaminated and clean items at both precisions")
    return float(
        (rates[(target, True)] - rates[(baseline, True)])
        - (rates[(target, False)] - rates[(baseline, False)])
    ) * 100.0


def aggregate_pilot(
    run_dir: str | Path,
    *,
    baseline: str = "bf16",
    target: str = "bnb_nf4",
) -> tuple[dict, dict]:
    """Read a completed pilot run and write its two registered JSON outputs."""
    run_dir = Path(run_dir)
    items, generations, scores = _read_tables(run_dir)
    greedy = generations[generations["is_greedy"]].copy()
    if "passed" not in greedy:
        greedy["passed"] = greedy["partial_pass_rate"].eq(1.0)
        pass_source = "derived_from_partial_pass_rate_eq_1"
    else:
        if greedy["passed"].isna().any():
            raise ValueError("greedy generation rows contain missing passed values")
        greedy["passed"] = greedy["passed"].astype(bool)
        pass_source = "generations.parquet:passed"

    models = sorted(set(greedy["model"]))
    labels = items.set_index("item_id")["contamination_proxy"].astype(bool)
    q1a: dict[str, dict] = {}
    q1b: dict[str, dict] = {}
    power_q1a: dict[str, dict] = {}
    power_q1b: dict[str, dict] = {}

    for model in models:
        q1a[model], q1b[model] = {}, {}
        power_q1a[model], power_q1b[model] = {}, {}
        for detector in PRIMARY_DETECTORS:
            paired = _paired_scores(scores, model, detector, baseline, target)
            paired_labels = labels.reindex(paired.index)
            if paired_labels.isna().any():
                raise ValueError(f"detector scores reference unknown items for {model}/{detector}")
            before, after = paired[baseline], paired[target]
            d = cohens_d_paired(before, after)
            auc_before = empirical_auc(before.to_numpy(), paired_labels.to_numpy())
            auc_after = empirical_auc(after.to_numpy(), paired_labels.to_numpy())
            r = pearson_r(before, after)
            q1a[model][detector] = {"cohens_d": d, "n_pairs": len(paired)}
            q1b[model][detector] = {
                "baseline_auc": auc_before, "target_auc": auc_after,
                "delta_auc": auc_after - auc_before, "cross_precision_r": r,
            }
            power_q1a[model][detector] = {
                "observed_abs_d": abs(d),
                "required_items": items_needed_paired_t(abs(d)) if d != 0 else None,
            }
            delta_auc = abs(auc_after - auc_before)
            power_q1b[model][detector] = {
                "observed_abs_delta_auc": delta_auc,
                "required_items_per_label_group": (
                    items_needed_for_delta_auc(delta_auc, auc_before, r)
                    if delta_auc > 0 and r < 1 else None
                ),
            }

    base_rates = {}
    for (model, quant, dataset), cell in greedy.merge(
        items[["item_id", "dataset"]], on="item_id"
    ).groupby(["model", "quant", "dataset"]):
        base_rates.setdefault(model, {}).setdefault(quant, {})[dataset] = base_rate(cell["passed"])

    q2_data = greedy.merge(items[["item_id", "contamination_proxy"]], on="item_id")
    q2_data = q2_data[q2_data["quant"].isin((baseline, target))].rename(
        columns={"quant": "precision", "contamination_proxy": "contaminated", "passed": "correct"}
    )
    q2_data["correct"] = q2_data["correct"].astype(int)
    glmm = fit_precision_contamination_glmm(q2_data)
    did_pp = _did_percentage_points(greedy, items, baseline, target)

    timing = {}
    for column in ("generation_seconds", "prompt_scoring_seconds", "sandbox_scoring_seconds"):
        if column in greedy and greedy[column].notna().any():
            values = greedy[column].dropna().astype(float)
            timing[column] = {
                "total": float(values.sum()), "mean_per_item": float(values.mean()),
                "items_per_second": float(len(values) / values.sum()) if values.sum() > 0 else None,
            }

    cdd_gates = {
        model: vars(check_cdd_gate(q1b[model]["cdd"]["baseline_auc"])) for model in models
    }
    summary = {
        "schema_version": 1, "baseline": baseline, "target": target,
        "models": models, "n_items": int(len(items)), "pass_at_1_source": pass_source,
        "q1a": q1a, "q1b": q1b,
        "q2": {"interaction_log_odds": glmm.interaction_log_odds,
               "interaction_sd": glmm.interaction_sd, "difference_in_differences_pp": did_pp,
               "item_level_r": pearson_r(
                   *[q2_data[q2_data["precision"] == q].sort_values(["model", "item_id"])["correct"]
                     for q in (baseline, target)]
               )},
        "base_rates": base_rates, "olmo3_proxy_label_error_rate": None,
        "olmo3_proxy_label_status": "pending_corpus_ground_truth",
        "cdd_gate": cdd_gates, "timing": timing,
    }
    power = {
        "schema_version": 1, "inputs_from": "pilot_summary.json",
        "q1a": power_q1a, "q1b": power_q1b,
        "q2": {"observed_abs_difference_in_differences_pp": abs(did_pp),
               "required_items_per_condition": (
                   items_needed_diff_in_diff(abs(did_pp)) if did_pp != 0 else None
               )},
    }
    for name, payload in (("pilot_summary.json", summary), ("power_recompute.json", power)):
        with (run_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    return summary, power
