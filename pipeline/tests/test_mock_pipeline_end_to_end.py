"""The pytest-runnable form of the local mock dry run (scripts/run_dry_run.py
is the CLI form; both call qcd.dry_run.run_dry_run()). Full step-1->9 code
path, zero GPU/downloads, asserting the pipeline's named invariants rather
than just "didn't crash":

- log-odds transform matches paper §4.5.3's worked table
- the HumanEval+MBPP+ pooling guard actually fires
- the mock's own expected-signed properties hold: contaminated items score
  higher on every detector and have a higher partial-pass rate than clean
  items (§2.4's peakedness/confidence intuition, exercised end-to-end)
"""

import dataclasses

from qcd.dry_run import run_dry_run


def test_dry_run_end_to_end_invariants(tmp_path):
    summary = run_dry_run(tmp_path, n_per_condition=4, n_cdd_samples=5, seed=0)

    assert summary.n_items == 16

    inv = summary.invariants
    assert inv.logodds_matches_paper_table is True
    assert inv.pooling_guard_fires is True
    assert all(inv.contaminated_scores_higher_than_clean.values()), inv.contaminated_scores_higher_than_clean
    assert inv.contaminated_has_higher_partial_pass is True


def test_dry_run_exposes_synthetic_diagnostics_only(tmp_path):
    summary = run_dry_run(tmp_path, n_per_condition=4, n_cdd_samples=5, seed=1)
    report = summary.diagnostics

    assert set(report.q1a_effect_size_d) == {"cdd", "perplexity", "mink_prob"}  # (a)
    assert set(report.q1b_baseline_auc) == {"cdd", "perplexity", "mink_prob"}  # (b)
    assert set(report.q1b_cross_precision_r) == {"cdd", "perplexity", "mink_prob"}
    assert report.q2_log_odds_effect == 0.0  # (c)
    assert report.q2_item_level_r == 1.0
    assert len(report.base_rates) == 4  # (d) one per Dataset condition


def test_dry_run_writes_raw_data_tree(tmp_path):
    summary = run_dry_run(tmp_path, n_per_condition=3, n_cdd_samples=3, seed=2)

    assert (summary.output_dir / "raw" / "items.parquet").exists()
    assert (summary.output_dir / "raw" / "generations.parquet").exists()
    assert (summary.output_dir / "raw" / "detector_scores.parquet").exists()


def test_dry_run_is_reproducible_given_same_seed(tmp_path):
    a = run_dry_run(tmp_path / "a", n_per_condition=3, n_cdd_samples=3, seed=99)
    b = run_dry_run(tmp_path / "b", n_per_condition=3, n_cdd_samples=3, seed=99)

    assert a.diagnostics.q1b_baseline_auc == b.diagnostics.q1b_baseline_auc


def test_dry_run_cli_main_exits_zero(capsys):
    from qcd.dry_run import main

    main()  # raises SystemExit(1) internally only on invariant failure

    captured = capsys.readouterr()
    assert "All invariant checks passed." in captured.out
