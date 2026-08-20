import json

import pandas as pd

from qcd.pilot.aggregate import aggregate_pilot


def test_aggregate_pilot_writes_registered_outputs(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    item_ids = [f"i{i}" for i in range(8)]
    labels = [False] * 4 + [True] * 4
    datasets = ["lcb_post"] * 4 + ["lcb_pre"] * 4
    pd.DataFrame({
        "item_id": item_ids, "dataset": datasets, "contamination_proxy": labels,
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
    assert summary["q1a"]["Qwen2.5-7B-Instruct"]["cdd"]["n_pairs"] == 8
    assert summary["olmo3_proxy_label_error_rate"] is None
    assert summary["timing"]["generation_seconds"]["total"] == 80.0
    assert "required_items" in power["q1a"]["Qwen2.5-7B-Instruct"]["cdd"]
    assert json.loads((tmp_path / "pilot_summary.json").read_text()) == summary
    assert json.loads((tmp_path / "power_recompute.json").read_text()) == power
