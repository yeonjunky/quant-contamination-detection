"""Development-only aggregation checks for item-level parquet data.

This module exercises analysis wiring while the final main-study analysis
command is still being validated. It is not called by dry runs or smoke tests,
does not make a CDD go/no-go decision, and its outputs are not manuscript
results or inputs to a data-dependent redesign.

**No sizing from observed effects.** Paper §4.7 ("Fixed study analysis
without a data-dependent pilot"): observed detector-score shifts, proxy AUCs
and base rates "do not trigger post-validation resizing or confirmatory
reclassification", and §4.6 forbids validation outputs from being aggregated
into "power estimates". An earlier version of this module computed
`required_items` from the observed d / ΔAUC / difference-in-differences and
wrote `development_power_diagnostics.json` in the same shape as a main-run
output (review finding E-F5). That whole path is gone: nothing here calls
`analysis.power`, and `aggregate_pilot` returns only the descriptive summary.
Planning numbers come from `analysis/power.py` with the paper's own
illustrative effect sizes, never from observed ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qcd.analysis.auc import empirical_auc
from qcd.analysis.conditional_logit import fit_conditional_logit_by_model
from qcd.analysis.mixed_effects import fit_precision_exposure_proxy_glmm
from qcd.data.schema import TemporalProxyLabel
from qcd.pilot.pilot_report import base_rate, cohens_d_paired, pearson_r

PRIMARY_DETECTORS = ("cdd", "perplexity", "mink_prob")
Q2_CONTRASTS = {
    "lcb_possible_vs_shared": "primary",
    "humaneval_vs_shared": "exploratory",
    "mbppplus_vs_shared": "exploratory",
}


def _read_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = run_dir / "raw"
    items_path = raw_dir / "items.parquet"
    labels_path = raw_dir / "model_item_labels.parquet"
    generation_paths = sorted(raw_dir.glob("generations*.parquet"))
    score_paths = sorted(raw_dir.glob("detector_scores*.parquet"))
    missing = []
    if not items_path.exists():
        missing.append(str(items_path))
    if not labels_path.exists():
        missing.append(str(labels_path))
    if not generation_paths:
        missing.append(str(raw_dir / "generations*.parquet"))
    if not score_paths:
        missing.append(str(raw_dir / "detector_scores*.parquet"))
    if missing:
        raise FileNotFoundError(f"missing development raw table(s): {', '.join(missing)}")
    return (
        pd.read_parquet(items_path),
        pd.read_parquet(labels_path),
        pd.concat((pd.read_parquet(path) for path in generation_paths), ignore_index=True),
        pd.concat((pd.read_parquet(path) for path in score_paths), ignore_index=True),
    )


def _validate_run_completeness(
    run_dir: Path,
    items: pd.DataFrame,
    labels: pd.DataFrame,
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
    expected_labels = len(config["models"]) * expected_items
    observed = (len(items), len(labels), len(generations), len(scores))
    expected = (expected_items, expected_labels, expected_generations, expected_scores)
    if observed != expected:
        raise ValueError(
            "development raw data are incomplete or contain stale parts: "
            f"observed items/labels/generations/scores={observed}, expected={expected}"
        )
    generation_key = ["model", "quant", "item_id", "sample_id"]
    score_key = ["model", "quant", "item_id", "detector"]
    if generations.duplicated(generation_key).any() or scores.duplicated(score_key).any():
        raise ValueError("development raw data contain duplicate generation or detector rows")


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
    rates = data.groupby(["quant", "exposure_proxy"])["passed"].mean()
    required = [(q, label) for q in (baseline, target) for label in (False, True)]
    if any(key not in rates.index for key in required):
        raise ValueError("Q2 needs possible-exposure and shared-control items at both precisions")
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


def _label_group_counts(data: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    unique = data[["model", "item_id", "exposure_proxy"]].drop_duplicates()
    for model, cell in unique.groupby("model"):
        counts[str(model)] = {
            "possible_exposure": int(cell["exposure_proxy"].sum()),
            "shared_control": int((~cell["exposure_proxy"]).sum()),
        }
    return counts


def _q2_base_rates(data: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    rates: dict[str, dict[str, dict[str, float]]] = {}
    for (model, quant, proxy), cell in data.groupby(
        ["model", "quant", "exposure_proxy"]
    ):
        group = "possible_exposure" if proxy else "shared_control"
        rates.setdefault(str(model), {}).setdefault(str(quant), {})[group] = base_rate(
            cell["passed"]
        )
    return rates


def _q2_cell(
    source: pd.DataFrame,
    contrast: str,
    label_column: str,
) -> pd.DataFrame:
    label = source[label_column]
    shared = label.eq(TemporalProxyLabel.SHARED_CLEAN_CONTROL.value)
    if contrast == "lcb_possible_vs_shared":
        possible = (
            source["dataset"].isin(("lcb_pre", "lcb_post"))
            & label.eq(TemporalProxyLabel.POSSIBLE_EXPOSURE.value)
        )
    elif contrast == "humaneval_vs_shared":
        possible = source["dataset"].eq("humaneval")
    elif contrast == "mbppplus_vs_shared":
        possible = source["dataset"].eq("mbppplus")
    else:  # pragma: no cover - guarded by the fixed Q2_CONTRASTS mapping
        raise ValueError(f"unknown Q2 contrast {contrast!r}")
    cell = source[possible | shared].copy()
    cell["exposure_proxy"] = possible[possible | shared].to_numpy()
    return cell


def _summarize_q2_cell(
    cell: pd.DataFrame,
    *,
    baseline: str,
    target: str,
    require_difficulty_check: bool,
) -> dict:
    if set(cell["exposure_proxy"]) != {False, True}:
        return {
            "status": "not_estimable_missing_proxy_group",
            "model_group_counts": _label_group_counts(cell),
        }
    paired_outcomes = _paired_outcomes(cell, baseline, target)
    complete_keys = set(paired_outcomes.index)
    cell = cell[
        cell.set_index(["model", "item_id"]).index.isin(complete_keys)
    ]
    q2_data = cell.rename(columns={"quant": "precision", "passed": "correct"})
    q2_data["correct"] = q2_data["correct"].astype(int)
    glmm = fit_precision_exposure_proxy_glmm(q2_data)
    # §4.5.5: the reported β_QE interval is the item-stratified conditional
    # logistic Wald interval, fitted separately per model — never the GLMM's
    # mean-field posterior SD.
    conditional_logit = {
        model: result.as_dict()
        for model, result in fit_conditional_logit_by_model(
            q2_data, baseline=baseline, target=target
        ).items()
    }
    did_pp = _did_percentage_points(cell, baseline, target)
    difficulty_did = {}
    if require_difficulty_check:
        for difficulty, stratum in cell.dropna(subset=["difficulty"]).groupby("difficulty"):
            if set(stratum["exposure_proxy"]) == {False, True}:
                difficulty_did[str(difficulty)] = _did_percentage_points(
                    stratum, baseline, target,
                )
        if not difficulty_did:
            raise ValueError(
                "the primary LCB Q2 contrast requires at least one shared native "
                "difficulty stratum"
            )
    result = {
        "status": "computed",
        "working_model_point_estimate": glmm.as_dict(),
        "beta_qe_interval_by_model": conditional_logit,
        "difference_in_differences_pp": did_pp,
        "item_level_r": pearson_r(paired_outcomes[baseline], paired_outcomes[target]),
        "n_pairs": len(paired_outcomes),
        "model_group_counts": _label_group_counts(cell),
        "base_rates_by_model": _q2_base_rates(cell),
        "difficulty_stratified_did_pp": difficulty_did,
        "difficulty_check_status": (
            "computed" if difficulty_did else "unavailable_no_shared_native_strata"
        ),
    }
    return result


def aggregate_pilot(
    run_dir: str | Path,
    *,
    baseline: str = "bf16",
    target: str = "bnb_nf4",
) -> dict:
    """Read completed development tables and write non-study diagnostics."""
    run_dir = Path(run_dir)
    items, labels, generations, scores = _read_tables(run_dir)
    _validate_run_completeness(run_dir, items, labels, generations, scores)
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
    required_label_columns = {
        "model", "item_id", "dataset", "primary_label", "sensitivity_label",
        "boundary_ambiguous", "publication_date", "primary_first_post_date",
        "sensitivity_first_post_date", "shared_control_start_date",
    }
    if not required_label_columns <= set(labels):
        missing = sorted(required_label_columns - set(labels))
        raise ValueError(f"model_item_labels.parquet is missing columns: {missing}")
    if labels.duplicated(["model", "item_id"]).any():
        raise ValueError("model_item_labels.parquet contains duplicate model-item rows")
    expected_label_keys = {
        (model, item_id) for model in models for item_id in items["item_id"]
    }
    observed_label_keys = set(zip(labels["model"], labels["item_id"]))
    if observed_label_keys != expected_label_keys:
        raise ValueError(
            "model_item_labels.parquet does not contain the exact model-item Cartesian product"
        )
    item_datasets = items.set_index("item_id")["dataset"]
    if not labels["dataset"].eq(labels["item_id"].map(item_datasets)).all():
        raise ValueError("model_item_labels.parquet dataset values disagree with items.parquet")
    q1a: dict[str, dict] = {}
    q1b: dict[str, dict] = {}

    for model in models:
        q1a[model], q1b[model] = {}, {}
        for detector in PRIMARY_DETECTORS:
            q1a[model][detector] = {}
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

            model_labels = labels[labels["model"] == model].set_index("item_id")
            lcb_labels = model_labels[
                model_labels["dataset"].isin(("lcb_pre", "lcb_post"))
            ]
            primary_labels = lcb_labels["primary_label"]
            primary_ids = set(primary_labels[
                primary_labels.isin((
                    TemporalProxyLabel.POSSIBLE_EXPOSURE.value,
                    TemporalProxyLabel.SHARED_CLEAN_CONTROL.value,
                ))
            ].index)
            paired = _paired_scores(
                scores, model, detector, baseline, target, primary_ids,
            )
            paired_label_values = primary_labels.reindex(paired.index)
            if paired_label_values.isna().any():
                raise ValueError(f"detector scores reference unknown items for {model}/{detector}")
            paired_labels = paired_label_values.eq(
                TemporalProxyLabel.POSSIBLE_EXPOSURE.value
            )
            if set(paired_labels) != {False, True}:
                raise ValueError(f"Q1b primary contrast lacks both proxy groups for {model}/{detector}")
            before, after = paired[baseline], paired[target]
            auc_before = empirical_auc(before.to_numpy(), paired_labels.to_numpy())
            auc_after = empirical_auc(after.to_numpy(), paired_labels.to_numpy())
            r = pearson_r(before, after)
            q1b[model][detector] = {
                "analysis_cell": "possible-exposure_vs_shared-clean-control",
                "baseline_auc": auc_before, "target_auc": auc_after,
                "delta_auc": auc_after - auc_before, "cross_precision_r": r,
                "n_pairs": len(paired),
                "n_possible_exposure": int(paired_labels.sum()),
                "n_shared_control": int((~paired_labels).sum()),
            }
            if bool(lcb_labels["boundary_ambiguous"].any()):
                sensitivity_labels = lcb_labels["sensitivity_label"]
                sensitivity_ids = set(sensitivity_labels[
                    sensitivity_labels.isin((
                        TemporalProxyLabel.POSSIBLE_EXPOSURE.value,
                        TemporalProxyLabel.SHARED_CLEAN_CONTROL.value,
                    ))
                ].index)
                sensitivity_paired = _paired_scores(
                    scores, model, detector, baseline, target, sensitivity_ids,
                )
                sensitivity_values = sensitivity_labels.reindex(sensitivity_paired.index)
                if sensitivity_values.isna().any():
                    raise ValueError(
                        f"sensitivity labels missing for {model}/{detector}"
                    )
                sensitivity_binary = sensitivity_values.eq(
                    TemporalProxyLabel.POSSIBLE_EXPOSURE.value
                )
                if set(sensitivity_binary) != {False, True}:
                    q1b[model][detector]["sensitivity"] = {
                        "status": "not_estimable_no_possible_exposure",
                        "n_possible_exposure": int(sensitivity_binary.sum()),
                        "n_shared_control": int((~sensitivity_binary).sum()),
                    }
                else:
                    sensitivity_before = sensitivity_paired[baseline]
                    sensitivity_after = sensitivity_paired[target]
                    q1b[model][detector]["sensitivity"] = {
                        "status": "computed",
                        "baseline_auc": empirical_auc(
                            sensitivity_before.to_numpy(), sensitivity_binary.to_numpy()
                        ),
                        "target_auc": empirical_auc(
                            sensitivity_after.to_numpy(), sensitivity_binary.to_numpy()
                        ),
                        "cross_precision_r": pearson_r(
                            sensitivity_before, sensitivity_after
                        ),
                        "n_pairs": len(sensitivity_paired),
                        "n_possible_exposure": int(sensitivity_binary.sum()),
                        "n_shared_control": int((~sensitivity_binary).sum()),
                    }

    base_rates = {}
    for (model, quant, dataset), cell in greedy.merge(
        items[["item_id", "dataset"]], on="item_id"
    ).groupby(["model", "quant", "dataset"]):
        base_rates.setdefault(model, {}).setdefault(quant, {})[dataset] = base_rate(cell["passed"])

    q2_source = greedy.merge(
        items[["item_id", "dataset", "difficulty"]], on="item_id"
    ).merge(
        labels[[
            "model", "item_id", "primary_label", "sensitivity_label", "boundary_ambiguous",
        ]],
        on=["model", "item_id"],
        validate="many_to_one",
    )
    q2_source = q2_source[q2_source["quant"].isin((baseline, target))]
    q2_results = {}
    for contrast, analysis_role in Q2_CONTRASTS.items():
        cell = _q2_cell(q2_source, contrast, "primary_label")
        result = _summarize_q2_cell(
            cell, baseline=baseline, target=target,
            require_difficulty_check=(contrast == "lcb_possible_vs_shared"),
        )
        result["analysis_role"] = analysis_role
        q2_results[contrast] = result
        if contrast == "lcb_possible_vs_shared" and bool(labels["boundary_ambiguous"].any()):
            sensitivity_cell = _q2_cell(q2_source, contrast, "sensitivity_label")
            sensitivity = _summarize_q2_cell(
                sensitivity_cell, baseline=baseline, target=target,
                require_difficulty_check=True,
            )
            sensitivity["analysis_role"] = "pre-specified sensitivity"
            q2_results[contrast]["sensitivity"] = sensitivity

    timing = {}
    for column in ("generation_seconds", "prompt_scoring_seconds", "sandbox_scoring_seconds"):
        if column in greedy and greedy[column].notna().any():
            values = greedy[column].dropna().astype(float)
            timing[column] = {
                "total": float(values.sum()), "mean_per_item": float(values.mean()),
                "items_per_second": float(len(values) / values.sum()) if values.sum() > 0 else None,
            }

    summary = {
        "schema_version": 5,
        "status": "development_only_not_manuscript_evidence",
        "sizing_from_observed_effects": "removed_see_paper_4_7",
        "baseline": baseline, "target": target,
        "models": models, "n_items": int(len(items)), "pass_at_1_source": pass_source,
        "q1a": q1a, "q1b": q1b,
        "q2": q2_results,
        "base_rates": base_rates,
        "timing": timing,
    }
    with (run_dir / "development_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
    # Remove a stale diagnostics file left by the pre-E-F5 version of this
    # module so a later reader cannot mistake it for current output.
    stale = run_dir / "development_power_diagnostics.json"
    if stale.exists():
        stale.unlink()
    return summary
