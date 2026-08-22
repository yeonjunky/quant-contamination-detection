import json

import pandas as pd
import pytest

from qcd.pilot.aggregate import aggregate_pilot


def test_aggregate_pilot_writes_registered_outputs(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    item_ids = [f"i{i}" for i in range(8)]
    labels = [False] * 4 + [True] * 4
    datasets = ["lcb_post"] * 4 + ["lcb_pre"] * 4
    pd.DataFrame({
        "item_id": item_ids, "dataset": datasets, "contamination_proxy": labels,
        "difficulty": ["easy", "medium", "hard", "easy"] * 2,
    }).to_parquet(raw / "items.parquet", index=False)

    generation_rows = []
    baseline_pass = [0, 1, 0, 1, 1, 1, 0, 1]
    target_pass = [0, 0, 0, 1, 1, 0, 0, 1]
    for model_index, model in enumerate(("Qwen2.5-7B-Instruct", "Olmo3-7B-Instruct")):
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
    baseline_scores = [0.2, 0.7, 0.4, 0.6, 0.9, 0.8, 0.3, 0.5]
    target_scores = [0.3, 0.65, 0.35, 0.55, 0.75, 0.7, 0.25, 0.45]
    for model in ("Qwen2.5-7B-Instruct", "Olmo3-7B-Instruct"):
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

    assert summary["n_items"] == 8
    assert summary["pass_at_1_source"] == "generations.parquet:passed"
    assert set(summary["q1a"]["Qwen2.5-7B-Instruct"]) == {"cdd", "perplexity", "mink_prob"}
    assert summary["q1a"]["Qwen2.5-7B-Instruct"]["cdd"]["lcb_pre"]["n_pairs"] == 4
    assert summary["q1b"]["Qwen2.5-7B-Instruct"]["cdd"]["n_pairs"] == 8
    assert set(summary["q2"]) == {"lcb_pre_vs_lcb_post"}
    assert summary["q2"]["lcb_pre_vs_lcb_post"]["difficulty_check_status"] == "computed"
    assert summary["olmo3_proxy_label_error_rate"] is None
    assert summary["timing"]["generation_seconds"]["total"] == 80.0
    assert "required_items" in power["q1a"]["Qwen2.5-7B-Instruct"]["cdd"]["lcb_pre"]
    assert json.loads((tmp_path / "pilot_summary.json").read_text()) == summary
    assert json.loads((tmp_path / "power_recompute.json").read_text()) == power


def test_aggregate_rejects_incomplete_manifest_run(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([
        {"item_id": "i1", "dataset": "lcb_pre", "contamination_proxy": True,
         "difficulty": "easy"},
    ]).to_parquet(raw / "items.parquet", index=False)
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
