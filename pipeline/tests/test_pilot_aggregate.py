import json

import pandas as pd
import pytest

from qcd.pilot.aggregate import aggregate_pilot


def test_development_aggregate_writes_non_study_diagnostics(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    item_ids = [f"i{i}" for i in range(10)]
    labels = [False] * 4 + [True] * 6
    datasets = ["lcb_post"] * 4 + ["lcb_pre"] * 4 + ["humaneval"] * 2
    pd.DataFrame({
        "item_id": item_ids, "dataset": datasets, "contamination_proxy": labels,
        "difficulty": ["easy", "medium", "hard", "easy"] * 2 + [None, None],
    }).to_parquet(raw / "items.parquet", index=False)
    models = ("Qwen2.5-7B-Instruct", "Olmo3-7B-Instruct")
    pd.DataFrame([
        {
            "model": model,
            "item_id": item_id,
            "dataset": dataset,
            "publication_date": "2024-01-01",
            "primary_first_post_date": "2025-01-01",
            "sensitivity_first_post_date": None,
            "shared_control_start_date": "2025-01-01",
            "primary_label": "possible-exposure" if proxy else "shared-clean-control",
            "sensitivity_label": "possible-exposure" if proxy else "shared-clean-control",
            "boundary_ambiguous": False,
        }
        for model in models
        for item_id, dataset, proxy in zip(item_ids, datasets, labels)
    ]).to_parquet(raw / "model_item_labels.parquet", index=False)

    generation_rows = []
    baseline_pass = [0, 1, 0, 1, 1, 1, 0, 1, 1, 0]
    target_pass = [0, 0, 0, 1, 1, 0, 0, 1, 1, 0]
    for model_index, model in enumerate(models):
        for quant, outcomes in (("bf16", baseline_pass), ("bnb_nf4", target_pass)):
            for index, (item_id, passed) in enumerate(zip(item_ids, outcomes)):
                generation_rows.append({
                    "model": model, "quant": quant, "item_id": item_id,
                    "sample_id": 0, "is_greedy": True,
                    "partial_pass_rate": float(passed), "passed": bool(passed),
                    "generation_seconds": 2.0 + model_index,
                    "prompt_scoring_seconds": 0.5,
                    "sandbox_scoring_seconds": 0.25,
                })
    pd.DataFrame(generation_rows).to_parquet(raw / "generations.parquet", index=False)

    score_rows = []
    baseline_scores = [0.2, 0.7, 0.4, 0.6, 0.9, 0.8, 0.3, 0.5, 0.95, 0.85]
    target_scores = [0.3, 0.65, 0.35, 0.55, 0.75, 0.7, 0.25, 0.45, 0.9, 0.8]
    for model in models:
        for detector_index, detector in enumerate(("cdd", "perplexity", "mink_prob")):
            offset = detector_index * 0.01
            for quant, values in (("bf16", baseline_scores), ("bnb_nf4", target_scores)):
                for item_id, value in zip(item_ids, values):
                    score_rows.append({
                        "model": model, "quant": quant, "item_id": item_id,
                        "detector": detector, "score": value + offset,
                    })
    pd.DataFrame(score_rows).to_parquet(raw / "detector_scores.parquet", index=False)

    summary, power = aggregate_pilot(tmp_path)

    assert summary["n_items"] == 10
    assert summary["pass_at_1_source"] == "generations.parquet:passed"
    assert set(summary["q1a"]["Qwen2.5-7B-Instruct"]) == {"cdd", "perplexity", "mink_prob"}
    assert summary["q1a"]["Qwen2.5-7B-Instruct"]["cdd"]["lcb_pre"]["n_pairs"] == 4
    q1b = summary["q1b"]["Qwen2.5-7B-Instruct"]["cdd"]
    assert q1b["n_pairs"] == 8
    assert q1b["n_possible_exposure"] == 4
    assert q1b["n_shared_control"] == 4
    assert set(summary["q2"]) == {
        "lcb_possible_vs_shared", "humaneval_vs_shared", "mbppplus_vs_shared",
    }
    assert summary["q2"]["lcb_possible_vs_shared"]["difficulty_check_status"] == "computed"
    assert summary["q2"]["lcb_possible_vs_shared"]["analysis_role"] == "primary"
    assert summary["q2"]["humaneval_vs_shared"]["analysis_role"] == "exploratory"
    assert set(
        summary["q2"]["lcb_possible_vs_shared"]["base_rates_by_model"]
        ["Qwen2.5-7B-Instruct"]["bf16"]
    ) == {"possible_exposure", "shared_control"}
    assert summary["schema_version"] == 4
    assert power["schema_version"] == 4
    assert summary["status"] == "development_only_not_manuscript_evidence"
    assert power["status"] == "development_only_not_study_resizing_input"
    assert "olmo3_proxy_label_error_rate" not in summary
    assert "cdd_gate" not in summary
    assert "c4_confirmatory_status" not in summary
    assert summary["timing"]["generation_seconds"]["total"] == 100.0
    assert "required_items" in power["q1a"]["Qwen2.5-7B-Instruct"]["cdd"]["lcb_pre"]
    assert json.loads((tmp_path / "development_summary.json").read_text()) == summary
    assert json.loads(
        (tmp_path / "development_power_diagnostics.json").read_text()
    ) == power


def test_aggregate_rejects_incomplete_manifest_run(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([
        {"item_id": "i1", "dataset": "lcb_pre", "contamination_proxy": True,
         "difficulty": "easy"},
    ]).to_parquet(raw / "items.parquet", index=False)
    pd.DataFrame([{
        "model": "model", "item_id": "i1", "dataset": "lcb_pre",
        "primary_label": "possible-exposure",
        "sensitivity_label": "possible-exposure",
        "boundary_ambiguous": False,
    }]).to_parquet(raw / "model_item_labels.parquet", index=False)
    pd.DataFrame([
        {"model": "model", "quant": "bf16", "item_id": "i1", "sample_id": 0,
         "is_greedy": True, "partial_pass_rate": 1.0, "passed": True},
    ]).to_parquet(raw / "generations.part.parquet", index=False)
    pd.DataFrame([
        {"model": "model", "quant": "bf16", "item_id": "i1",
         "detector": "cdd", "score": 0.5},
    ]).to_parquet(raw / "detector_scores.part.parquet", index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "config": {
            "models": ["model"], "quant_levels": ["bf16"], "n_items": 1,
            "n_cdd_samples": 1,
        },
    }))

    with pytest.raises(ValueError, match="incomplete or contain stale parts"):
        aggregate_pilot(tmp_path)
