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
Q2_CONTRASTS = {
    "lcb_pre_vs_lcb_post": ("lcb_pre", "lcb_post"),
    "humaneval_vs_lcb_post": ("humaneval", "lcb_post"),
    "mbppplus_vs_lcb_post": ("mbppplus", "lcb_post"),
}


def _read_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = run_dir / "raw"
    items_path = raw_dir / "items.parquet"
    generation_paths = sorted(raw_dir.glob("generations*.parquet"))
    score_paths = sorted(raw_dir.glob("detector_scores*.parquet"))
    missing = []
    if not items_path.exists():
        missing.append(str(items_path))
    if not generation_paths:
        missing.append(str(raw_dir / "generations*.parquet"))
    if not score_paths:
        missing.append(str(raw_dir / "detector_scores*.parquet"))
    if missing:
        raise FileNotFoundError(f"missing pilot raw table(s): {', '.join(missing)}")
    return (
        pd.read_parquet(items_path),
        pd.concat((pd.read_parquet(path) for path in generation_paths), ignore_index=True),
        pd.concat((pd.read_parquet(path) for path in score_paths), ignore_index=True),
    )


def _validate_run_completeness(
    run_dir: Path,
    items: pd.DataFrame,
    generations: pd.DataFrame,
    scores: pd.DataFrame,
) -> None:
    """Reject an interrupted real run before producing inferential summaries."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return  # Allows analysis of imported/legacy tables without a run manifest.
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config")
    if not config:
        raise ValueError("manifest.json does not contain the executed configuration")
    expected_items = int(config["n_items"])
    expected_cells = len(config["models"]) * len(config["quant_levels"]) * expected_items
    expected_generations = expected_cells * (int(config["n_cdd_samples"]) + 1)
    expected_scores = expected_cells * 5
    observed = (len(items), len(generations), len(scores))
    expected = (expected_items, expected_generations, expected_scores)
    if observed != expected:
        raise ValueError(
            "pilot raw data are incomplete or contain stale parts: "
            f"observed items/generations/scores={observed}, expected={expected}"
        )
    generation_key = ["model", "quant", "item_id", "sample_id"]
    score_key = ["model", "quant", "item_id", "detector"]
    if generations.duplicated(generation_key).any() or scores.duplicated(score_key).any():
        raise ValueError("pilot raw data contain duplicate generation or detector rows")


def _paired_scores(
    scores: pd.DataFrame, model: str, detector: str, baseline: str, target: str,
    item_ids: set[str],
) -> pd.DataFrame:
    subset = scores[
        (scores["model"] == model)
        & (scores["detector"] == detector)
        & scores["item_id"].isin(item_ids)
    ]
    paired = subset.pivot(index="item_id", columns="quant", values="score")
    if baseline not in paired or target not in paired:
        raise ValueError(f"missing {baseline}/{target} scores for {model}/{detector}")
    return paired[[baseline, target]].dropna()


def _did_percentage_points(data: pd.DataFrame, baseline: str, target: str) -> float:
    rates = data.groupby(["quant", "contaminated"])["passed"].mean()
    required = [(q, label) for q in (baseline, target) for label in (False, True)]
    if any(key not in rates.index for key in required):
        raise ValueError("Q2 needs contaminated and clean items at both precisions")
    return float(
        (rates[(target, True)] - rates[(baseline, True)])
        - (rates[(target, False)] - rates[(baseline, False)])
    ) * 100.0


def _paired_outcomes(data: pd.DataFrame, baseline: str, target: str) -> pd.DataFrame:
    paired = data.pivot(index=["model", "item_id"], columns="quant", values="passed")
    if baseline not in paired or target not in paired:
        raise ValueError(f"Q2 is missing {baseline}/{target} outcomes")
    paired = paired[[baseline, target]].dropna()
    if paired.empty:
        raise ValueError("Q2 has no complete item-level precision pairs")
    return paired


def aggregate_pilot(
    run_dir: str | Path,
    *,
    baseline: str = "bf16",
    target: str = "bnb_nf4",
) -> tuple[dict, dict]:
    """Read a completed pilot run and write its two registered JSON outputs."""
    run_dir = Path(run_dir)
    items, generations, scores = _read_tables(run_dir)
    _validate_run_completeness(run_dir, items, generations, scores)
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
    if items["item_id"].duplicated().any():
        raise ValueError("items.parquet contains duplicate item_id values")
    item_table = items.set_index("item_id")
    q1a: dict[str, dict] = {}
    q1b: dict[str, dict] = {}
    power_q1a: dict[str, dict] = {}
    power_q1b: dict[str, dict] = {}

    for model in models:
        q1a[model], q1b[model] = {}, {}
        power_q1a[model], power_q1b[model] = {}, {}
        for detector in PRIMARY_DETECTORS:
            q1a[model][detector], power_q1a[model][detector] = {}, {}
            for dataset, cell_items in items.groupby("dataset"):
                paired = _paired_scores(
                    scores, model, detector, baseline, target,
                    set(cell_items["item_id"]),
                )
                if len(paired) < 2:
                    continue
                before, after = paired[baseline], paired[target]
                d = cohens_d_paired(before, after)
                q1a[model][detector][dataset] = {
                    "cohens_d": d, "n_pairs": len(paired),
                }
                power_q1a[model][detector][dataset] = {
                    "observed_abs_d": abs(d),
                    "required_items": items_needed_paired_t(abs(d)) if d != 0 else None,
                }

            lcb_items = items[items["dataset"].isin(("lcb_pre", "lcb_post"))]
            paired = _paired_scores(
                scores, model, detector, baseline, target, set(lcb_items["item_id"]),
            )
            paired_labels = item_table["contamination_proxy"].reindex(paired.index)
            if paired_labels.isna().any():
                raise ValueError(f"detector scores reference unknown items for {model}/{detector}")
            before, after = paired[baseline], paired[target]
            auc_before = empirical_auc(before.to_numpy(), paired_labels.to_numpy())
            auc_after = empirical_auc(after.to_numpy(), paired_labels.to_numpy())
            r = pearson_r(before, after)
            q1b[model][detector] = {
                "analysis_cell": "lcb_pre_vs_lcb_post",
                "baseline_auc": auc_before, "target_auc": auc_after,
                "delta_auc": auc_after - auc_before, "cross_precision_r": r,
                "n_pairs": len(paired),
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

    q2_source = greedy.merge(
        items[["item_id", "dataset", "difficulty"]], on="item_id"
    )
    q2_source = q2_source[q2_source["quant"].isin((baseline, target))]
    q2_results = {}
    power_q2 = {}
    for contrast, (suspect, control) in Q2_CONTRASTS.items():
        if not {suspect, control} <= set(q2_source["dataset"]):
            continue
        cell = q2_source[q2_source["dataset"].isin((suspect, control))].copy()
        cell["contaminated"] = cell["dataset"].eq(suspect)
        paired_outcomes = _paired_outcomes(cell, baseline, target)
        complete_keys = set(paired_outcomes.index)
        cell = cell[
            cell.set_index(["model", "item_id"]).index.isin(complete_keys)
        ]
        q2_data = cell.rename(columns={"quant": "precision", "passed": "correct"})
        q2_data["correct"] = q2_data["correct"].astype(int)
        glmm = fit_precision_contamination_glmm(q2_data)
        did_pp = _did_percentage_points(cell, baseline, target)
        difficulty_did = {}
        if contrast == "lcb_pre_vs_lcb_post":
            for difficulty, stratum in cell.dropna(subset=["difficulty"]).groupby("difficulty"):
                if set(stratum["contaminated"]) == {False, True}:
                    difficulty_did[str(difficulty)] = _did_percentage_points(
                        stratum, baseline, target,
                    )
            if not difficulty_did:
                raise ValueError(
                    "the primary LCB Q2 contrast requires at least one shared native "
                    "difficulty stratum"
                )
        q2_results[contrast] = {
            "interaction_log_odds": glmm.interaction_log_odds,
            "interaction_sd": glmm.interaction_sd,
            "difference_in_differences_pp": did_pp,
            "item_level_r": pearson_r(
                paired_outcomes[baseline], paired_outcomes[target]
            ),
            "n_pairs": len(paired_outcomes),
            "difficulty_stratified_did_pp": difficulty_did,
            "difficulty_check_status": (
                "computed" if difficulty_did else "unavailable_no_shared_native_strata"
            ),
        }
        power_q2[contrast] = {
            "observed_abs_difference_in_differences_pp": abs(did_pp),
            "required_items_per_condition": (
                items_needed_diff_in_diff(abs(did_pp)) if did_pp != 0 else None
            ),
        }

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
        "schema_version": 2, "baseline": baseline, "target": target,
        "models": models, "n_items": int(len(items)), "pass_at_1_source": pass_source,
        "q1a": q1a, "q1b": q1b,
        "q2": q2_results,
        "base_rates": base_rates, "olmo3_proxy_label_error_rate": None,
        "olmo3_proxy_label_status": "pending_corpus_ground_truth",
        "cdd_gate": cdd_gates, "timing": timing,
    }
    power = {
        "schema_version": 2, "inputs_from": "pilot_summary.json",
        "q1a": power_q1a, "q1b": power_q1b,
        "q2": power_q2,
    }
    for name, payload in (("pilot_summary.json", summary), ("power_recompute.json", power)):
        with (run_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    return summary, power
