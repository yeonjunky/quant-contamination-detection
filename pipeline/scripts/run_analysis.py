#!/usr/bin/env python
"""Frozen analysis of a main-study raw tree — paper §4.5.6's confirmatory
family, §4.5.5's per-model β_QE intervals, and §4.4's truncated-generation
rate by precision.

What this driver runs, and nothing else:

1. **§4.5.6's four confirmatory tests**, fixed to Qwen2.5-32B-Instruct and the
   bf16 -> BNB-nf4 contrast, Holm-corrected at familywise α=0.05. C1-C3 are
   the paired nf4-bf16 shift of perplexity, Min-k% Prob and CDD over all of
   that model's LiveCodeBench items; C4 is the probability-family-minus-CDD
   AUC rank-reversal test on the `possible-exposure` / `shared-clean-control`
   items. The family is four slots — the model, the contrast, the detectors
   and the item sets are command-line arguments nowhere, and a
   not-estimable slot is retained with p=1 rather than dropped.
2. **§4.5.5's β_QE interval**, from an item-stratified conditional logistic
   fit run separately per model, one quantized level against bf16 at a time.
   Both β_QE and J=−β_QE are reported with their intervals.
3. **§4.4's truncated-generation rate by precision**, because "a truncation
   rate that differs across precisions would confound a pass@1 shift with a
   length-cap artifact".

Two gates run before any of that:

- `require_main_study()` on the input tree (paper §4.6 — engineering
  validation output is never manuscript evidence). This is the first thing
  the script does, before it opens a single parquet file.
- the §4.5.5 interval-coverage record must exist and be complete, because the
  analysis manifest has to carry its generating parameters, replication count
  and achieved coverage. The check "fixes which method supplies the reported
  interval"; an analysis that cannot show it has no reported interval.

Usage:
    python scripts/run_analysis.py ../data/raw/main
    python scripts/run_analysis.py ../data/raw/main --out /somewhere/analysis

Outputs (in `--out`, default `<run_dir>/analysis`):
    confirmatory_family.json   C1-C4, raw and Holm-adjusted p-values
    beta_qe_intervals.json     per-model conditional-logit β_QE and J
    truncation_rates.json      §4.4 rates by precision
    analysis_manifest.json     what §4.5.5/§4.5.6 require to be recorded
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:  # allow running without an editable install
    sys.path.insert(0, str(REPO_SRC))

from qcd.analysis import study_inputs  # noqa: E402
from qcd.analysis.conditional_logit import (  # noqa: E402
    DEFAULT_INTERVAL_LEVEL,
    fit_conditional_logit_by_model,
)
from qcd.analysis.confirmatory import (  # noqa: E402
    CONFIRMATORY_SLOTS,
    Q1A_SLOT_DETECTORS,
    assemble_confirmatory_family,
    c4_rank_reversal_test,
    paired_score_shift_test,
)
from qcd.analysis.coverage import (  # noqa: E402
    INTERVAL_COVERAGE_COMMAND,
    INTERVAL_COVERAGE_RECORD_PATH,
    interval_coverage_record_digest,
    load_interval_coverage_record,
)
from qcd.config import Quant  # noqa: E402
from qcd.constants import ALPHA  # noqa: E402
from qcd.io.manifest import (  # noqa: E402
    StudyPhase,
    build_manifest,
    require_main_study,
    write_manifest,
)
from qcd.models.registry import QWEN2_5_32B  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]

# §4.5.6: "The four confirmatory tests are restricted to Qwen2.5-32B-Instruct,
# the Primary model in §4.1, and the bf16->BNB-nf4 contrast." Module
# constants, not CLI flags: the family is fixed before main-study outcomes and
# no invocation of this script may point it at another arm.
CONFIRMATORY_MODEL = QWEN2_5_32B.name
CONFIRMATORY_BASELINE = Quant.BF16.value
CONFIRMATORY_TARGET = Quant.BNB_NF4.value

CONFIRMATORY_FAMILY_FILENAME = "confirmatory_family.json"
BETA_QE_FILENAME = "beta_qe_intervals.json"
TRUNCATION_FILENAME = "truncation_rates.json"
ANALYSIS_MANIFEST_FILENAME = "analysis_manifest.json"


# --------------------------------------------------------------------------
# §4.5.6 — the confirmatory family
# --------------------------------------------------------------------------


def run_confirmatory(tables: study_inputs.RawTables) -> dict:
    """C1-C4 on the fixed arm, with the item accounting behind each test."""
    paired = study_inputs.paired_detector_scores(
        tables,
        model=CONFIRMATORY_MODEL,
        baseline=CONFIRMATORY_BASELINE,
        target=CONFIRMATORY_TARGET,
    )
    q1a_results = {}
    for slot, detector in Q1A_SLOT_DETECTORS.items():
        pair = paired.arrays.get(detector)
        if pair is None:
            # A detector with no scores at all still occupies its slot; the
            # test itself reports the not-estimable status and p=1 (§4.5.6).
            q1a_results[slot] = paired_score_shift_test([], [], detector=detector)
        else:
            q1a_results[slot] = paired_score_shift_test(pair[0], pair[1], detector=detector)

    c4_data = study_inputs.c4_inputs(
        tables,
        model=CONFIRMATORY_MODEL,
        baseline=CONFIRMATORY_BASELINE,
        target=CONFIRMATORY_TARGET,
    )
    c4 = c4_rank_reversal_test(
        c4_data.baseline_scores,
        c4_data.target_scores,
        c4_data.labels,
        baseline=CONFIRMATORY_BASELINE,
        target=CONFIRMATORY_TARGET,
    )
    family = assemble_confirmatory_family(
        q1a_results,
        c4,
        model=CONFIRMATORY_MODEL,
        baseline=CONFIRMATORY_BASELINE,
        target=CONFIRMATORY_TARGET,
        familywise_alpha=ALPHA,
    )
    payload = family.as_dict()
    payload["slot_detectors"] = dict(Q1A_SLOT_DETECTORS)
    payload["item_accounting"] = {
        "c1_c3": {
            "sample": "all LiveCodeBench items of the confirmatory model (§4.5.6)",
            "n_lcb_items": paired.n_lcb_items,
            "n_paired_items": paired.n_complete_items,
            "n_items_dropped_incomplete": paired.n_items_dropped_incomplete,
            "detectors_with_no_scores": paired.detectors_missing,
        },
        "c4": {
            "sample": "`possible-exposure` vs. `shared-clean-control` LiveCodeBench items (§4.5.6)",
            "label_field": c4_data.label_field,
            "n_possible_exposure": c4_data.n_possible_exposure,
            "n_shared_clean_control": c4_data.n_shared_clean_control,
            "n_items_dropped_incomplete": c4_data.n_items_dropped_incomplete,
        },
    }
    return payload


# --------------------------------------------------------------------------
# §4.5.5 — per-model β_QE intervals
# --------------------------------------------------------------------------


def run_beta_qe_intervals(
    tables: study_inputs.RawTables, *, interval_level: float = DEFAULT_INTERVAL_LEVEL
) -> dict:
    """§4.5.5's reported interval: item-stratified conditional logistic Wald,
    "run separately per model", one quantized level against bf16 at a time."""
    frame = study_inputs.outcome_frame(tables)
    baseline = Quant.BF16.value
    present = sorted(set(frame["precision"]) - {baseline})
    contrasts = {}
    for target in present:
        cell = frame[frame["precision"].isin((baseline, target))]
        if cell.empty:
            continue
        fits = fit_conditional_logit_by_model(
            cell, baseline=baseline, target=target, interval_level=interval_level
        )
        contrasts[f"{baseline}->{target}"] = {
            model: result.as_dict() for model, result in fits.items()
        }
    return {
        "model_formula": "correct ~ precision * exposure_proxy + (1 | item) + (1 | model) (§4.5.5)",
        "reported_interval": (
            "item-stratified conditional logistic Wald interval for β_QE, fitted per model "
            "(§4.5.5); the mean-field VB posterior SD is explicitly not used"
        ),
        "coding": (
            "§3.1: Q=0 bf16, Q=1 the quantized level under test; E=0 `shared-clean-control`, "
            "E=1 `possible-exposure`; regressors Q and Q×E; J = −β_QE, and negating the "
            "interval reverses its endpoints"
        ),
        "outcome": "greedy pass@1 (`generations.passed`), §4.4",
        "interval_level": interval_level,
        "n_rows": int(frame.shape[0]),
        "contrasts": contrasts,
    }


# --------------------------------------------------------------------------
# The analysis manifest (§4.5.5's recording requirement)
# --------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_analysis_manifest(
    *,
    run_dir: Path,
    input_manifest: dict,
    tables: study_inputs.RawTables,
    coverage_record: dict,
    coverage_path: Path,
    outputs: dict[str, Path],
    interval_level: float,
):
    """Everything §4.5.5 and §4.5.6 require to be recorded next to the
    numbers: the interval method, the coverage check that fixed it, the
    library versions, the input run's identity and the code commit."""
    coverage_summary = [
        {
            "true_beta_qe": scenario["design"].get("beta_qe"),
            "n_replications": scenario["n_replications"],
            "seed": scenario.get("seed"),
            "generating_parameters": scenario["design"],
            "achieved_coverage": {
                method: {
                    "coverage": result.get("coverage"),
                    "coverage_monte_carlo_se": result.get("coverage_monte_carlo_se"),
                    "n_estimable": result.get("n_estimable"),
                    "n_replications": result.get("n_replications"),
                    "mean_interval_width": result.get("mean_interval_width"),
                }
                for method, result in scenario["coverage"].items()
            },
        }
        for scenario in coverage_record["scenarios"]
    ]
    analysis_config = {
        "driver": "scripts/run_analysis.py",
        "confirmatory_family": {
            "slots": list(CONFIRMATORY_SLOTS),
            "model": CONFIRMATORY_MODEL,
            "baseline": CONFIRMATORY_BASELINE,
            "target": CONFIRMATORY_TARGET,
            "slot_detectors": dict(Q1A_SLOT_DETECTORS),
            "familywise_alpha": ALPHA,
            "multiplicity": "Holm, familywise α=0.05 over the four slots (§4.5.6)",
        },
        "interval_method": "item_stratified_conditional_logit_wald",
        "interval_level": interval_level,
        "label_field": "primary_label",
        "input_config_hash": input_manifest.get("config_hash"),
        "input_study_phase": input_manifest.get("study_phase"),
        "interval_coverage_record_sha256": interval_coverage_record_digest(coverage_path),
    }
    return build_manifest(
        analysis_config,
        study_phase=StudyPhase.MAIN_STUDY,
        repo_dir=_REPO_ROOT,
        extra={
            "record": "paper_4_5_5_4_5_6_analysis_manifest",
            "input_run": {
                "path": str(run_dir),
                "raw_dir": str(tables.raw_dir),
                "study_phase": input_manifest.get("study_phase"),
                "config_hash": input_manifest.get("config_hash"),
                "git_commit": input_manifest.get("git_commit"),
                "timestamp_utc": input_manifest.get("timestamp_utc"),
                "source_files": tables.source_files,
            },
            "interval": {
                "reported_method": "item_stratified_conditional_logit_wald",
                "interval_level": interval_level,
                "ruled_out": {
                    "variational_bayes_posterior_sd": (
                        "§4.5.5: a mean-field VB posterior SD discards the coefficients' "
                        "correlations and does not carry its nominal coverage; it may still "
                        "supply point estimates and variance components, but not interval width"
                    )
                },
                "permitted_alternatives": [
                    "MCMC from the full posterior",
                    "Laplace approximation inverting the full joint Hessian",
                ],
                "priors": (
                    "none — the adopted interval comes from a conditional likelihood, which "
                    "has no prior. The ruled-out comparator in the coverage check used "
                    "statsmodels BinomialBayesMixedGLM.from_formula's defaults (vcp_p=1, "
                    "fe_p=2)"
                ),
                "coding": (
                    "§3.1: Q=0 bf16 / Q=1 quantized level under test; E=0 `shared-clean-control` "
                    "/ E=1 `possible-exposure`; regressors Q and Q×E; J = −β_QE"
                ),
            },
            "interval_coverage_check": {
                "path": str(coverage_path.relative_to(_REPO_ROOT))
                if coverage_path.is_relative_to(_REPO_ROOT)
                else str(coverage_path),
                "sha256": interval_coverage_record_digest(coverage_path),
                "regenerate_with": INTERVAL_COVERAGE_COMMAND,
                "nominal_level": coverage_record.get("nominal_level"),
                "adopted_interval_method": coverage_record.get("adopted_interval_method"),
                "methods_examined": coverage_record.get("methods_examined"),
                "library_versions": coverage_record.get("library_versions"),
                "scenarios": coverage_summary,
            },
            "outputs": {
                name: {"file": path.name, "sha256": _sha256_file(path)}
                for name, path in sorted(outputs.items())
            },
        },
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="main-study run directory (the one holding manifest.json and raw/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="analysis output directory (default: <run_dir>/analysis)",
    )
    parser.add_argument(
        "--coverage-record",
        type=Path,
        default=INTERVAL_COVERAGE_RECORD_PATH,
        help="§4.5.5 interval-coverage record (default: %(default)s)",
    )
    parser.add_argument(
        "--interval-level",
        type=float,
        default=DEFAULT_INTERVAL_LEVEL,
        help="two-sided β_QE interval level (default: %(default)s)",
    )
    return parser


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def run_analysis(
    run_dir: Path,
    *,
    out_dir: Path | None = None,
    coverage_record_path: Path = INTERVAL_COVERAGE_RECORD_PATH,
    interval_level: float = DEFAULT_INTERVAL_LEVEL,
) -> dict[str, Path]:
    """The whole analysis, as a function so tests can drive it directly."""
    # Paper §4.6 first, before anything is read: validation output is never
    # manuscript evidence.
    input_manifest = require_main_study(run_dir, consumer="scripts/run_analysis.py")
    # §4.5.5: the coverage check fixes which method supplies the reported
    # interval, and the analysis manifest has to carry it.
    coverage_record = load_interval_coverage_record(coverage_record_path)

    tables = study_inputs.load_raw_tables(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else Path(run_dir) / "analysis"

    outputs = {
        "confirmatory_family": _write_json(
            out_dir / CONFIRMATORY_FAMILY_FILENAME, run_confirmatory(tables)
        ),
        "beta_qe_intervals": _write_json(
            out_dir / BETA_QE_FILENAME,
            run_beta_qe_intervals(tables, interval_level=interval_level),
        ),
        "truncation_rates": _write_json(
            out_dir / TRUNCATION_FILENAME, study_inputs.truncated_generation_rates(tables)
        ),
    }
    manifest = build_analysis_manifest(
        run_dir=Path(run_dir),
        input_manifest=input_manifest,
        tables=tables,
        coverage_record=coverage_record,
        coverage_path=Path(coverage_record_path),
        outputs=outputs,
        interval_level=interval_level,
    )
    outputs["analysis_manifest"] = write_manifest(
        manifest, out_dir / ANALYSIS_MANIFEST_FILENAME
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = run_analysis(
        args.run_dir,
        out_dir=args.out,
        coverage_record_path=args.coverage_record,
        interval_level=args.interval_level,
    )
    family = json.loads(outputs["confirmatory_family"].read_text(encoding="utf-8"))
    print(f"Confirmatory family (§4.5.6) — {CONFIRMATORY_MODEL}, "
          f"{CONFIRMATORY_BASELINE}->{CONFIRMATORY_TARGET}, Holm at α={ALPHA}:")
    for slot in CONFIRMATORY_SLOTS:
        print(
            f"  {slot}: status={family['tests'][slot]['status']} "
            f"p={family['raw_p_values'][slot]:.4g} "
            f"holm={family['holm_adjusted_p_values'][slot]:.4g} "
            f"rejected={family['rejected'][slot]}"
        )
    for name, path in sorted(outputs.items()):
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
