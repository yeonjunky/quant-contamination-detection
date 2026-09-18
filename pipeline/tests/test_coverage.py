"""Interval-coverage utility — paper §4.5.5's pre-main-run check.

These tests check the *machinery* (generator shape, shared replications
across methods, coverage arithmetic, not-estimable bookkeeping) and one
cheap end-to-end run. The study's actual coverage numbers come from
`scripts/verify_interval_coverage.py` at the replication count §4.5.5
requires, not from the test suite.
"""

import numpy as np
import pytest

from qcd.analysis.coverage import (
    DESIGN_N_POSSIBLE_EXPOSURE,
    DESIGN_N_SHARED_CLEAN_CONTROL,
    INTERVAL_METHODS,
    CoverageResult,
    SyntheticDesign,
    conditional_logit_wald,
    estimate_interval_coverage,
    simulate_design_dataset,
)
from qcd.analysis.logodds import marginal_accuracy
from qcd.constants import BASE_RATE_LCB_POST_ILLUSTRATIVE, DIFFICULTY_SIGMA


def test_default_design_is_the_qwen_lcb_label_group_shape():
    """§4.5.6: 690 `possible-exposure` and 182 `shared-clean-control`; §4.5.5
    asks for "a normal item-difficulty distribution"."""
    design = SyntheticDesign()
    assert (design.n_possible_exposure, design.n_shared_clean_control) == (690, 182)
    assert (DESIGN_N_POSSIBLE_EXPOSURE, DESIGN_N_SHARED_CLEAN_CONTROL) == (690, 182)
    assert design.difficulty_sd == DIFFICULTY_SIGMA


def test_simulated_frame_has_the_design_shape():
    design = SyntheticDesign()
    df = simulate_design_dataset(design, np.random.default_rng(0))

    n_items = design.n_possible_exposure + design.n_shared_clean_control
    assert len(df) == 2 * n_items
    assert set(df["precision"]) == {"bf16", "bnb_nf4"}
    per_item = df.groupby("item_id")["precision"].nunique()
    assert (per_item == 2).all()
    exposure_by_item = df.drop_duplicates("item_id")["exposure_proxy"]
    assert int(exposure_by_item.sum()) == design.n_possible_exposure
    assert int((~exposure_by_item).sum()) == design.n_shared_clean_control
    # Exposure is an item-level label: constant across precision within item.
    assert (df.groupby("item_id")["exposure_proxy"].nunique() == 1).all()


def test_simulated_base_rate_matches_the_requested_marginal_accuracy():
    """The intercept is solved with `analysis.logodds`, so the baseline
    (Q=0, E=0) cell reproduces the requested marginal accuracy."""
    design = SyntheticDesign(
        beta_q=0.0, beta_e=0.0, beta_qe=0.0,
        n_possible_exposure=0, n_shared_clean_control=40_000,
    )
    df = simulate_design_dataset(design, np.random.default_rng(7))

    analytic = marginal_accuracy(design.intercept, design.difficulty_sd)
    assert analytic == pytest.approx(BASE_RATE_LCB_POST_ILLUSTRATIVE, abs=1e-6)
    assert df["correct"].mean() == pytest.approx(BASE_RATE_LCB_POST_ILLUSTRATIVE, abs=0.01)


def test_simulated_beta_qe_is_recovered_by_the_adopted_method():
    design = SyntheticDesign(beta_qe=0.9)
    df = simulate_design_dataset(design, np.random.default_rng(21))

    point, low, high = conditional_logit_wald(df, design)

    assert low < point < high
    assert point == pytest.approx(0.9, abs=0.6)


def test_design_as_dict_records_the_generating_parameters():
    """§4.5.5: "Record the generating parameters, the number of replications,
    and the achieved coverage of every method examined"."""
    payload = SyntheticDesign(beta_qe=0.25).as_dict()
    assert payload["beta_qe"] == 0.25
    assert payload["difficulty_sd"] == DIFFICULTY_SIGMA
    assert payload["n_possible_exposure"] == 690
    assert payload["intercept"] == pytest.approx(SyntheticDesign().intercept)


def test_every_method_sees_the_same_replications():
    calls = {"a": [], "b": []}

    def recorder(key):
        def method(df, design):
            calls[key].append(int(df["correct"].sum()))
            return (0.0, -1.0, 1.0)

        return method

    design = SyntheticDesign(n_possible_exposure=30, n_shared_clean_control=10, beta_qe=0.0)
    estimate_interval_coverage(
        design=design,
        n_replications=5,
        methods={"a": recorder("a"), "b": recorder("b")},
        seed=3,
    )
    assert calls["a"] == calls["b"]
    assert len(calls["a"]) == 5


def test_coverage_arithmetic_and_not_estimable_bookkeeping():
    true_beta = 0.5

    def always_covers(df, design):
        return (true_beta, true_beta - 0.1, true_beta + 0.1)

    def never_covers(df, design):
        return (true_beta, true_beta + 1.0, true_beta + 2.0)

    def half_estimable(df, design):
        half_estimable.calls += 1
        if half_estimable.calls % 2:
            return None
        return (true_beta, true_beta - 0.1, true_beta + 0.1)

    half_estimable.calls = 0

    design = SyntheticDesign(
        n_possible_exposure=20, n_shared_clean_control=10, beta_qe=true_beta
    )
    out = estimate_interval_coverage(
        design=design,
        n_replications=10,
        methods={
            "always": always_covers,
            "never": never_covers,
            "half": half_estimable,
        },
        seed=1,
    )

    assert out["always"].coverage == 1.0
    assert out["always"].n_covered == 10
    assert out["always"].mean_interval_width == pytest.approx(0.2)
    assert out["never"].coverage == 0.0
    assert out["half"].n_estimable == 5
    assert out["half"].coverage == 1.0
    # Monte Carlo SE uses the estimable denominator.
    assert out["half"].coverage_monte_carlo_se == pytest.approx(0.0)


def test_coverage_result_with_no_estimable_replication():
    design = SyntheticDesign(n_possible_exposure=10, n_shared_clean_control=5)
    out = estimate_interval_coverage(
        design=design,
        n_replications=3,
        methods={"none": lambda df, d: None},
        seed=0,
    )
    result = out["none"]
    assert isinstance(result, CoverageResult)
    assert result.n_estimable == 0
    assert result.coverage is None
    assert result.mean_interval_width is None


def test_monte_carlo_se_matches_the_binomial_formula():
    true_beta = 0.0
    flip = {"n": 0}

    def alternating(df, design):
        flip["n"] += 1
        if flip["n"] % 2:
            return (0.0, -1.0, 1.0)  # covers
        return (0.0, 2.0, 3.0)  # misses

    design = SyntheticDesign(
        n_possible_exposure=20, n_shared_clean_control=10, beta_qe=true_beta
    )
    result = estimate_interval_coverage(
        design=design, n_replications=20, methods={"alt": alternating}, seed=0
    )["alt"]

    assert result.coverage == pytest.approx(0.5)
    assert result.coverage_monte_carlo_se == pytest.approx(np.sqrt(0.5 * 0.5 / 20))


def test_registry_holds_the_adopted_and_the_ruled_out_method():
    """§4.5.5 adopts the conditional-logit Wald interval and rules out the
    mean-field VB posterior SD; both stay in the comparison so the ruling is
    measured, not asserted."""
    assert "conditional_logit_wald" in INTERVAL_METHODS
    assert "variational_bayes_posterior_sd" in INTERVAL_METHODS


def test_small_end_to_end_run_of_the_adopted_method():
    design = SyntheticDesign(
        n_possible_exposure=200, n_shared_clean_control=60, beta_qe=0.0
    )
    result = estimate_interval_coverage(
        design=design,
        n_replications=25,
        methods={"conditional_logit_wald": conditional_logit_wald},
        seed=17,
    )["conditional_logit_wald"]

    assert result.n_estimable == 25
    assert result.coverage >= 0.80  # loose: 25 replications, nominal 0.95
    assert result.mean_interval_width > 0
    assert result.mean_point_estimate == pytest.approx(0.0, abs=0.4)
