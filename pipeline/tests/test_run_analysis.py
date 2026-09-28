"""The analysis entry point — paper §4.6's gate, §4.5.6's confirmatory family,
§4.5.5's β_QE intervals and manifest requirements, and §4.4's
truncated-generation rate.

The synthetic run directory below is written with the **real**
`io/raw_writer.py` and the real `data/temporal_labels.py`, so a column the
analysis reads by name has to exist there too; a schema drift breaks these
tests rather than silently producing an empty analysis.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from qcd.analysis.confirmatory import holm_adjusted_p_values
from qcd.analysis.coverage import (
    INTERVAL_COVERAGE_RECORD_PATH,
    load_interval_coverage_record,
)
from qcd.config import Quant
from qcd.data.schema import Dataset, Item
from qcd.data.temporal_labels import materialize_model_item_labels
from qcd.io.manifest import StudyPhase, build_manifest, write_manifest
from qcd.io.raw_writer import RawDataWriter
from qcd.models.registry import QWEN2_5_7B, QWEN2_5_32B

_SCRIPTS = Path(__file__).parents[1] / "scripts"

_BOUNDARY = dt.datetime.fromisoformat("2025-01-01")
_MODELS = (QWEN2_5_32B, QWEN2_5_7B)
_DETECTORS = ("perplexity", "mink_prob", "cdd")

# Item counts of the synthetic tree, shaped like §4.5.6's three LCB groups
# (possible-exposure / intermediate / shared-clean-control) at 1/20th scale.
N_POSSIBLE_EXPOSURE = 30
N_INTERMEDIATE = 8
N_SHARED_CONTROL = 16
N_LCB_ITEMS = N_POSSIBLE_EXPOSURE + N_INTERMEDIATE + N_SHARED_CONTROL


def _load_run_analysis():
    spec = importlib.util.spec_from_file_location("run_analysis", _SCRIPTS / "run_analysis.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synthetic_items() -> list[Item]:
    """LCB items on three sides of the two boundaries Qwen2.5-32B-Instruct
    uses: its own 2024-09-20 first-post date and the shared 2025-01-01
    control boundary."""
    items: list[Item] = []
    for index in range(N_POSSIBLE_EXPOSURE):
        items.append(
            Item(
                item_id=f"pre_{index:04d}",
                dataset=Dataset.LCB_PRE,
                prompt=f"possible-exposure problem {index}",
                release_version="release_v6",
                metadata={"contest_date": f"2024-0{1 + index % 8}-15T00:00:00"},
            )
        )
    for index in range(N_INTERMEDIATE):
        items.append(
            Item(
                item_id=f"mid_{index:04d}",
                dataset=Dataset.LCB_PRE,
                prompt=f"intermediate-date problem {index}",
                release_version="release_v6",
                metadata={"contest_date": f"2024-1{index % 2}-05T00:00:00"},
            )
        )
    for index in range(N_SHARED_CONTROL):
        items.append(
            Item(
                item_id=f"post_{index:04d}",
                dataset=Dataset.LCB_POST,
                prompt=f"shared-clean-control problem {index}",
                release_version="release_v6",
                metadata={"contest_date": f"2025-0{1 + index % 6}-10T00:00:00"},
            )
        )
    return items


def _item_base_scores(rng, *, exposed: bool) -> dict[str, float]:
    """One item's bf16 detector scores, oriented as `qcd.detectors` stores
    them (larger = more possible exposure)."""
    exposure_bump = 0.8 if exposed else 0.0
    return {
        "perplexity": -1.5 + exposure_bump + rng.normal(0, 0.4),
        "mink_prob": -2.0 + exposure_bump + rng.normal(0, 0.4),
        "cdd": 0.30 + (0.05 if exposed else 0.0) + rng.normal(0, 0.05),
    }


def _arm_scores(base: dict[str, float], *, quant: str, rng) -> dict[str, float]:
    """The same item at one precision. The two arms share the item's base
    score — the real design measures one item at both precisions, and C1-C3
    are paired tests — and nf4 adds a real perplexity shift, a null Min-k
    shift and a small CDD shift, so C1/C2/C3 do not all land the same way."""
    if quant == Quant.BF16.value:
        return dict(base)
    return {
        "perplexity": base["perplexity"] + 0.30 + rng.normal(0, 0.05),
        "mink_prob": base["mink_prob"] + rng.normal(0, 0.05),
        "cdd": base["cdd"] - 0.02 + rng.normal(0, 0.01),
    }


def _write_run(
    tmp_path: Path,
    *,
    study_phase: StudyPhase = StudyPhase.MAIN_STUDY,
    seed: int = 7,
) -> Path:
    rng = np.random.default_rng(seed)
    items = _synthetic_items()
    run_dir = tmp_path / "run"
    writer = RawDataWriter(run_dir / "raw")
    writer.write_items(items)
    labels = materialize_model_item_labels(items, _MODELS, shared_control_boundary=_BOUNDARY)
    writer.write_model_item_labels(labels)
    label_lookup = {(row["model"], row["item_id"]): row["primary_label"] for row in labels}

    for model in _MODELS:
        base_scores = {
            item.item_id: _item_base_scores(
                rng, exposed=label_lookup[(model.name, item.item_id)] == "possible-exposure"
            )
            for item in items
        }
        for quant in (Quant.BF16.value, Quant.BNB_NF4.value):
            for index, item in enumerate(items):
                label = label_lookup[(model.name, item.item_id)]
                exposed = label == "possible-exposure"
                # Outcome: quantization drops accuracy, and drops it further
                # on possible-exposure items, so β_QE is identified.
                probability = 0.75 if not exposed else 0.80
                if quant == Quant.BNB_NF4.value:
                    probability -= 0.10 + (0.20 if exposed else 0.0)
                passed = bool(rng.random() < probability)
                # One flag left unwritten per model/precision, to exercise the
                # §4.4 "flag missing" accounting.
                truncated = None if index == 0 else bool(index % 10 == 0)
                writer.add_generation(
                    model=model.name,
                    quant=quant,
                    item_id=item.item_id,
                    sample_id=0,
                    is_greedy=True,
                    text="print(1)",
                    token_ids=[1, 2],
                    token_logprobs=[-0.1, -0.2],
                    prompt_token_logprobs=[-0.3, -0.4],
                    passed=passed,
                    partial_pass_rate=1.0 if passed else 0.0,
                    truncated_at_cap=truncated,
                    max_new_tokens=512,
                )
                writer.add_generation(
                    model=model.name,
                    quant=quant,
                    item_id=item.item_id,
                    sample_id=1,
                    is_greedy=False,
                    text="print(2)",
                    token_ids=[1, 3],
                    token_logprobs=[-0.1, -0.5],
                    truncated_at_cap=bool(index % 5 == 0),
                    max_new_tokens=512,
                )
                scores = _arm_scores(base_scores[item.item_id], quant=quant, rng=rng)
                for detector, score in scores.items():
                    writer.add_detector_score(
                        model=model.name,
                        quant=quant,
                        item_id=item.item_id,
                        detector=detector,
                        score=score,
                    )
                # §4.4's separate completion-based diagnostics: written by
                # real_run.py, used by no confirmatory test.
                writer.add_detector_score(
                    model=model.name, quant=quant, item_id=item.item_id,
                    detector="completion_perplexity", score=float("nan"),
                )
            writer.flush(part=f"{model.name}-{quant}")

    write_manifest(
        build_manifest(
            {"driver": "tests/test_run_analysis.py", "synthetic": True},
            study_phase=study_phase,
            seed=seed,
        ),
        run_dir / "manifest.json",
    )
    return run_dir


@pytest.fixture(scope="module")
def run_analysis_module():
    return _load_run_analysis()


@pytest.fixture(scope="module")
def analysis_outputs(tmp_path_factory, run_analysis_module):
    run_dir = _write_run(tmp_path_factory.mktemp("main_study"))
    outputs = run_analysis_module.run_analysis(run_dir)
    return {
        name: json.loads(path.read_text(encoding="utf-8")) for name, path in outputs.items()
    } | {"_paths": outputs, "_run_dir": run_dir}


# --- (a) the §4.6 gate ------------------------------------------------------


def test_analysis_refuses_engineering_validation_output(tmp_path, run_analysis_module):
    run_dir = _write_run(tmp_path, study_phase=StudyPhase.ENGINEERING_VALIDATION)
    with pytest.raises(ValueError, match="engineering_validation"):
        run_analysis_module.run_analysis(run_dir)
    # Refused before anything was written, not after.
    assert not (run_dir / "analysis").exists()


def test_analysis_refuses_a_tree_without_a_manifest(tmp_path, run_analysis_module):
    run_dir = _write_run(tmp_path)
    (run_dir / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="no run manifest"):
        run_analysis_module.run_analysis(run_dir)


def test_the_gate_runs_before_the_coverage_record_is_read(tmp_path, run_analysis_module):
    # Order matters: a validation tree must be refused for being validation
    # output, not for a missing artifact.
    run_dir = _write_run(tmp_path, study_phase=StudyPhase.ENGINEERING_VALIDATION)
    with pytest.raises(ValueError, match="engineering_validation"):
        run_analysis_module.run_analysis(
            run_dir, coverage_record_path=tmp_path / "does-not-exist.json"
        )


def test_a_missing_coverage_record_stops_the_analysis(tmp_path, run_analysis_module):
    run_dir = _write_run(tmp_path)
    with pytest.raises(FileNotFoundError, match="interval-coverage record"):
        run_analysis_module.run_analysis(
            run_dir, coverage_record_path=tmp_path / "does-not-exist.json"
        )


# --- (b) the four confirmatory tests and Holm -------------------------------


def test_family_has_exactly_the_four_slots(analysis_outputs):
    family = analysis_outputs["confirmatory_family"]
    assert family["slots"] == ["C1", "C2", "C3", "C4"]
    assert sorted(family["tests"]) == ["C1", "C2", "C3", "C4"]
    assert family["model"] == QWEN2_5_32B.name
    assert (family["baseline"], family["target"]) == ("bf16", "bnb_nf4")
    assert family["slot_detectors"] == {"C1": "perplexity", "C2": "mink_prob", "C3": "cdd"}


def test_every_slot_is_estimable_on_the_synthetic_tree(analysis_outputs):
    family = analysis_outputs["confirmatory_family"]
    assert [family["tests"][slot]["status"] for slot in ("C1", "C2", "C3", "C4")] == [
        "computed"
    ] * 4


def test_holm_matches_the_module_and_is_applied_to_all_four(analysis_outputs):
    family = analysis_outputs["confirmatory_family"]
    raw = [family["raw_p_values"][slot] for slot in ("C1", "C2", "C3", "C4")]
    expected = holm_adjusted_p_values(raw)
    got = [family["holm_adjusted_p_values"][slot] for slot in ("C1", "C2", "C3", "C4")]
    assert got == pytest.approx(list(expected))
    assert family["familywise_alpha"] == 0.05
    assert all(
        family["rejected"][slot] == (family["holm_adjusted_p_values"][slot] <= 0.05)
        for slot in ("C1", "C2", "C3", "C4")
    )


def test_the_planted_shifts_come_back_with_the_right_signs(analysis_outputs):
    tests = analysis_outputs["confirmatory_family"]["tests"]
    # +0.30 planted on perplexity, none on Min-k, -0.02 on CDD.
    assert tests["C1"]["mean_difference"] == pytest.approx(0.30, abs=0.03)
    assert tests["C2"]["mean_difference"] == pytest.approx(0.0, abs=0.03)
    assert tests["C3"]["mean_difference"] == pytest.approx(-0.02, abs=0.01)
    assert tests["C1"]["p_value"] < 0.01
    assert tests["C2"]["p_value"] > 0.05


def test_c1_c3_use_every_lcb_item_and_c4_only_the_labelled_two_groups(analysis_outputs):
    accounting = analysis_outputs["confirmatory_family"]["item_accounting"]
    # §4.5.6: C1-C3 run on all LCB items, including the intermediate-date ones.
    assert accounting["c1_c3"]["n_lcb_items"] == N_LCB_ITEMS
    assert accounting["c1_c3"]["n_paired_items"] == N_LCB_ITEMS
    assert accounting["c1_c3"]["n_items_dropped_incomplete"] == 0
    for slot in ("C1", "C2", "C3"):
        assert analysis_outputs["confirmatory_family"]["tests"][slot]["n_pairs"] == N_LCB_ITEMS
    # C4 excludes the intermediate `clean-by-model-cutoff` items.
    c4 = analysis_outputs["confirmatory_family"]["tests"]["C4"]
    assert c4["n_possible_exposure"] == N_POSSIBLE_EXPOSURE
    assert c4["n_shared_clean_control"] == N_SHARED_CONTROL
    assert accounting["c4"]["label_field"] == "primary_label"


def test_c4_reports_all_six_component_aucs(analysis_outputs):
    component = analysis_outputs["confirmatory_family"]["tests"]["C4"]["component_aucs"]
    assert set(component) == {
        f"{detector}@{quant}"
        for detector in _DETECTORS
        for quant in ("bf16", "bnb_nf4")
    }


def test_missing_scores_keep_the_slot_with_p_one(tmp_path, run_analysis_module):
    """§4.5.6: a not-estimable slot is retained with p=1 for multiplicity
    accounting, never dropped or replaced."""
    import pandas as pd

    run_dir = _write_run(tmp_path)
    for path in (run_dir / "raw").glob("detector_scores.*.parquet"):
        frame = pd.read_parquet(path)
        frame = frame[frame["detector"] != "cdd"]
        frame.to_parquet(path, index=False)
    outputs = run_analysis_module.run_analysis(run_dir)
    family = json.loads(outputs["confirmatory_family"].read_text(encoding="utf-8"))
    assert sorted(family["tests"]) == ["C1", "C2", "C3", "C4"]
    assert family["raw_p_values"]["C3"] == 1.0
    assert family["raw_p_values"]["C4"] == 1.0
    assert family["tests"]["C4"]["status"].startswith("not_estimable")
    assert family["rejected"]["C3"] is False


# --- (c) the analysis manifest ---------------------------------------------


def test_manifest_records_the_input_run_identity(analysis_outputs):
    manifest = analysis_outputs["analysis_manifest"]
    input_run = manifest["extra"]["input_run"]
    source = json.loads(
        (analysis_outputs["_run_dir"] / "manifest.json").read_text(encoding="utf-8")
    )
    assert input_run["study_phase"] == "main_study" == source["study_phase"]
    assert input_run["config_hash"] == source["config_hash"]
    assert manifest["config"]["input_config_hash"] == source["config_hash"]
    assert manifest["config"]["input_study_phase"] == "main_study"


def test_manifest_records_the_interval_method_and_what_is_ruled_out(analysis_outputs):
    interval = analysis_outputs["analysis_manifest"]["extra"]["interval"]
    assert interval["reported_method"] == "item_stratified_conditional_logit_wald"
    assert "variational_bayes_posterior_sd" in interval["ruled_out"]
    assert interval["interval_level"] == 0.95


def test_manifest_carries_the_coverage_check_in_full(analysis_outputs):
    """§4.5.5: "Record the generating parameters, the number of replications,
    and the achieved coverage of every method examined in the analysis
    manifest."""
    check = analysis_outputs["analysis_manifest"]["extra"]["interval_coverage_check"]
    record = load_interval_coverage_record(INTERVAL_COVERAGE_RECORD_PATH)
    assert check["path"].endswith("analysis_artifacts/interval_coverage_check.json")
    assert check["sha256"]
    assert check["methods_examined"] == record["methods_examined"]
    assert check["scenarios"]
    for scenario in check["scenarios"]:
        assert scenario["n_replications"] >= 1
        # generating parameters, verbatim
        for key in ("n_possible_exposure", "n_shared_clean_control", "difficulty_sd", "beta_qe"):
            assert key in scenario["generating_parameters"]
        # achieved coverage of *every* method examined
        assert set(scenario["achieved_coverage"]) == set(record["methods_examined"])
        for method in record["methods_examined"]:
            assert scenario["achieved_coverage"][method]["coverage"] is not None


def test_manifest_records_library_versions_and_the_code_commit(analysis_outputs):
    manifest = analysis_outputs["analysis_manifest"]
    versions = manifest["package_versions"]
    for package in ("scipy", "statsmodels", "numpy", "pandas"):
        assert versions[package], f"{package} version missing from the analysis manifest"
    assert "git_commit" in manifest
    assert manifest["study_phase"] == "main_study"


def test_manifest_lists_its_outputs_with_digests(analysis_outputs):
    outputs = analysis_outputs["analysis_manifest"]["extra"]["outputs"]
    assert set(outputs) == {"confirmatory_family", "beta_qe_intervals", "truncation_rates"}
    for entry in outputs.values():
        assert len(entry["sha256"]) == 64


def test_manifest_names_the_fixed_confirmatory_family(analysis_outputs):
    family = analysis_outputs["analysis_manifest"]["config"]["confirmatory_family"]
    assert family["slots"] == ["C1", "C2", "C3", "C4"]
    assert family["model"] == QWEN2_5_32B.name
    assert family["target"] == "bnb_nf4"
    assert family["familywise_alpha"] == 0.05


# --- §4.5.5's per-model β_QE intervals --------------------------------------


def test_beta_qe_is_reported_per_model_with_j_and_a_reversed_interval(analysis_outputs):
    payload = analysis_outputs["beta_qe_intervals"]
    contrast = payload["contrasts"]["bf16->bnb_nf4"]
    assert set(contrast) == {QWEN2_5_32B.name, QWEN2_5_7B.name}
    for result in contrast.values():
        assert result["status"] == "computed"
        assert result["interval_method"] == "item_stratified_conditional_logit_wald"
        assert result["ci_low"] < result["beta_qe"] < result["ci_high"]
        # §4.5.5: J = −β_QE, and negating an interval reverses its endpoints.
        assert result["j"] == pytest.approx(-result["beta_qe"])
        assert result["j_ci_low"] == pytest.approx(-result["ci_high"])
        assert result["j_ci_high"] == pytest.approx(-result["ci_low"])


def test_beta_qe_input_excludes_intermediate_items(analysis_outputs):
    # §3.1's 2x2 has two exposure cells; `clean-by-model-cutoff` is neither.
    payload = analysis_outputs["beta_qe_intervals"]
    expected = len(_MODELS) * 2 * (N_POSSIBLE_EXPOSURE + N_SHARED_CONTROL)
    assert payload["n_rows"] == expected


# --- (d) §4.4's truncated-generation rate by precision ----------------------


def test_truncation_rate_is_reported_by_precision(analysis_outputs):
    rates = analysis_outputs["truncation_rates"]
    precisions = {row["precision"] for row in rates["by_precision"]}
    assert precisions == {"bf16", "bnb_nf4"}
    assert rates["max_new_tokens_observed"] == [512]
    assert rates["n_greedy_generations"] == len(_MODELS) * 2 * N_LCB_ITEMS


def test_truncation_counts_match_the_planted_flags(analysis_outputs):
    rates = {row["precision"]: row for row in analysis_outputs["truncation_rates"]["by_precision"]}
    # Per model/precision the greedy rows plant: index 0 -> flag missing,
    # every 10th index -> truncated (indices 10, 20, 30, 40, 50).
    planted_truncated = len([i for i in range(1, N_LCB_ITEMS) if i % 10 == 0])
    for precision in ("bf16", "bnb_nf4"):
        greedy = rates[precision]["greedy_only"]
        assert greedy["n_flag_missing"] == len(_MODELS)
        assert greedy["n_truncated"] == planted_truncated * len(_MODELS)
        assert greedy["truncated_rate"] == pytest.approx(
            greedy["n_truncated"] / (greedy["n_generations"] - greedy["n_flag_missing"])
        )
        # The sampled rows are counted too, but separately from pass@1's own rate.
        assert rates[precision]["all_generations"]["n_generations"] == (
            2 * len(_MODELS) * N_LCB_ITEMS
        )


def test_truncation_is_also_broken_out_per_model(analysis_outputs):
    rows = analysis_outputs["truncation_rates"]["by_model_and_precision"]
    assert {(row["model"], row["precision"]) for row in rows} == {
        (model.name, precision)
        for model in _MODELS
        for precision in ("bf16", "bnb_nf4")
    }


# --- part files a resumed run may have written twice ------------------------


def test_an_identical_duplicate_part_file_does_not_double_count(tmp_path, run_analysis_module):
    """A resumed run can flush the same batch twice; identical rows are a
    duplicate, not two measurements."""
    import shutil

    run_dir = _write_run(tmp_path)
    for path in sorted((run_dir / "raw").glob("detector_scores.*.parquet")):
        shutil.copy(path, path.with_name(path.name.replace(".parquet", "-copy.parquet")))
    for path in sorted((run_dir / "raw").glob("generations.*.parquet")):
        shutil.copy(path, path.with_name(path.name.replace(".parquet", "-copy.parquet")))
    outputs = run_analysis_module.run_analysis(run_dir)
    family = json.loads(outputs["confirmatory_family"].read_text(encoding="utf-8"))
    assert family["tests"]["C1"]["n_pairs"] == N_LCB_ITEMS
    rates = json.loads(outputs["truncation_rates"].read_text(encoding="utf-8"))
    assert rates["n_greedy_generations"] == len(_MODELS) * 2 * N_LCB_ITEMS


def test_conflicting_duplicate_rows_are_refused(tmp_path, run_analysis_module):
    import pandas as pd

    run_dir = _write_run(tmp_path)
    path = sorted((run_dir / "raw").glob("detector_scores.*.parquet"))[0]
    frame = pd.read_parquet(path)
    conflicting = frame.head(1).copy()
    conflicting["score"] = conflicting["score"] + 1.0
    pd.concat([frame, conflicting], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="different 'score' values"):
        run_analysis_module.run_analysis(run_dir)


# --- the coverage record lives in the repository ----------------------------


def test_the_coverage_record_is_a_tracked_repository_artifact():
    assert INTERVAL_COVERAGE_RECORD_PATH.exists(), (
        "the §4.5.5 coverage record must live in the repository, not a temp directory"
    )
    record = load_interval_coverage_record()
    assert record["adopted_interval_method"] == "conditional_logit_wald"
    assert set(record["methods_examined"]) >= {
        "conditional_logit_wald",
        "variational_bayes_posterior_sd",
    }


def test_the_coverage_script_writes_the_tracked_path_by_default():
    spec = importlib.util.spec_from_file_location(
        "verify_interval_coverage", _SCRIPTS / "verify_interval_coverage.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    # The record's default destination is the tracked artifact, so a re-run
    # refreshes the file the analysis manifest cites.
    assert module.INTERVAL_COVERAGE_RECORD_PATH == INTERVAL_COVERAGE_RECORD_PATH
