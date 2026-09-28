"""Numerical validation of the four confirmatory tests C1-C4 (paper §4.5.6).

Each block states what it is cross-checked against:

- paired t-test -> `scipy.stats.ttest_rel` (the independent reference the
  task specification names). This checks the wiring that actually risks being
  wrong: which array is subtracted from which, the degrees of freedom, and
  two-sidedness.
- DeLong covariance -> `analysis.auc.hanley_mcneil_se_unequal` for the
  diagonal (same order of magnitude on synthetic data), the exact
  `empirical_auc` estimator for the AUC point estimates, and algebraic
  identities for the off-diagonal (a duplicated score column must give a
  perfectly correlated, zero-variance difference).
- Holm -> `statsmodels.stats.multitest.multipletests(method="holm")`.
- Reversal rule -> all four sign combinations plus the zero boundary.
"""

import numpy as np
import pytest

from qcd.analysis.auc import empirical_auc, hanley_mcneil_se_unequal
from qcd.analysis.confirmatory import (
    CONFIRMATORY_SLOTS,
    STATUS_COMPUTED,
    STATUS_EMPTY_LABEL_GROUP,
    STATUS_INSUFFICIENT_PAIRS,
    STATUS_MISSING_SCORES,
    STATUS_ZERO_VARIANCE,
    C4Result,
    assemble_confirmatory_family,
    c4_p_value,
    c4_rank_reversal_test,
    delong_auc_covariance,
    holm_adjusted_p_values,
    holm_decisions,
    is_reversal,
    paired_score_shift_test,
    run_confirmatory_family,
)


# --------------------------------------------------------------------------
# C1-C3
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_paired_shift_matches_scipy_ttest_rel(seed):
    from scipy import stats

    rng = np.random.default_rng(seed)
    before = rng.normal(0.0, 1.0, 400)
    after = before + rng.normal(0.15, 0.9, 400)

    result = paired_score_shift_test(before, after, detector="perplexity")
    reference = stats.ttest_rel(after, before)

    assert result.status == STATUS_COMPUTED
    assert result.n_pairs == 400
    assert result.degrees_of_freedom == 399
    assert result.t_statistic == pytest.approx(float(reference.statistic), rel=1e-12)
    assert result.p_value == pytest.approx(float(reference.pvalue), rel=1e-12)


def test_paired_shift_estimand_is_target_minus_baseline():
    """§4.5.6: the estimand is the mean within-item nf4 - bf16 difference."""
    baseline = np.array([1.0, 2.0, 3.0, 4.0])
    target = np.array([1.5, 2.5, 3.0, 5.0])

    result = paired_score_shift_test(baseline, target, detector="cdd")

    assert result.mean_difference == pytest.approx(np.mean(target - baseline))
    assert result.mean_difference > 0
    flipped = paired_score_shift_test(target, baseline, detector="cdd")
    assert flipped.mean_difference == pytest.approx(-result.mean_difference)
    assert flipped.p_value == pytest.approx(result.p_value)


def test_paired_shift_cohens_dz_is_mean_over_sd_of_differences():
    """§4.5.1: "Cohen's d here is d_z, the mean difference divided by the
    standard deviation of item differences"."""
    rng = np.random.default_rng(3)
    before = rng.normal(size=200)
    after = before + rng.normal(0.3, 1.0, 200)

    result = paired_score_shift_test(before, after, detector="mink_prob")
    diff = after - before

    assert result.cohens_dz == pytest.approx(diff.mean() / diff.std(ddof=1))
    # d_z relates to t by t = d_z * sqrt(n)
    assert result.t_statistic == pytest.approx(result.cohens_dz * np.sqrt(200))


def test_paired_shift_zero_variance_is_not_estimable_with_p_one():
    result = paired_score_shift_test(
        np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]), detector="cdd"
    )
    assert result.status == STATUS_ZERO_VARIANCE
    assert result.p_value == 1.0
    assert result.t_statistic is None
    # The constant shift itself is still reported, only the test is not.
    assert result.mean_difference == pytest.approx(1.0)


def test_paired_shift_missing_or_too_few_scores_is_not_estimable():
    missing = paired_score_shift_test(
        np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 4.0]), detector="cdd"
    )
    assert missing.status == STATUS_MISSING_SCORES
    assert missing.p_value == 1.0

    too_few = paired_score_shift_test(np.array([1.0]), np.array([2.0]), detector="cdd")
    assert too_few.status == STATUS_INSUFFICIENT_PAIRS
    assert too_few.p_value == 1.0


# --------------------------------------------------------------------------
# DeLong
# --------------------------------------------------------------------------


def _synthetic_auc_scores(rng, n_pos, n_neg, separation, n_detectors=1, shared=0.0):
    labels = np.concatenate([np.ones(n_pos, dtype=bool), np.zeros(n_neg, dtype=bool)])
    common = rng.normal(size=n_pos + n_neg)
    rows = []
    for _ in range(n_detectors):
        noise = rng.normal(size=n_pos + n_neg)
        score = shared * common + np.sqrt(max(1 - shared**2, 0.0)) * noise
        rows.append(score + separation * labels)
    return np.vstack(rows), labels


def test_delong_point_estimates_match_empirical_auc():
    rng = np.random.default_rng(11)
    scores, labels = _synthetic_auc_scores(rng, 200, 120, 0.8, n_detectors=3, shared=0.5)

    result = delong_auc_covariance(scores, labels, names=("a", "b", "c"))

    for index, name in enumerate(("a", "b", "c")):
        assert result.auc(name) == pytest.approx(empirical_auc(scores[index], labels))
    assert result.n_positive == 200
    assert result.n_negative == 120


def test_delong_se_is_same_order_as_hanley_mcneil():
    """Hanley-McNeil is a parametric (binormal-ish) approximation and DeLong
    is nonparametric, so they are not expected to agree to many digits — the
    check is that they agree to within a factor of ~1.3 on data generated
    close to the assumptions."""
    rng = np.random.default_rng(23)
    n_pos, n_neg = 690, 182
    scores, labels = _synthetic_auc_scores(rng, n_pos, n_neg, 0.75, n_detectors=1)

    result = delong_auc_covariance(scores, labels, names=("only",))
    delong_se = float(np.sqrt(result.covariance[0, 0]))
    hm_se = hanley_mcneil_se_unequal(result.auc("only"), n_pos, n_neg)

    assert 0.75 < delong_se / hm_se < 1.35
    assert delong_se == pytest.approx(hm_se, abs=0.01)


def test_delong_identical_estimates_give_singular_covariance_and_zero_gap_variance():
    """Two copies of the same score vector must have unit correlation, so the
    difference contrast has exactly zero variance. §4.5.6 notes that such a
    singular six-AUC covariance does not invalidate the scalar contrasts —
    no inverse is taken."""
    rng = np.random.default_rng(5)
    scores, labels = _synthetic_auc_scores(rng, 150, 90, 0.6, n_detectors=1)
    doubled = np.vstack([scores[0], scores[0]])

    result = delong_auc_covariance(doubled, labels, names=("x", "y"))

    assert result.aucs[0] == pytest.approx(result.aucs[1])
    assert result.covariance[0, 1] == pytest.approx(result.covariance[0, 0])
    estimate, se = result.contrast({"x": 1.0, "y": -1.0})
    assert estimate == pytest.approx(0.0, abs=1e-12)
    assert se == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.matrix_rank(result.covariance) == 1


def test_delong_covariance_is_symmetric_positive_semidefinite():
    rng = np.random.default_rng(31)
    scores, labels = _synthetic_auc_scores(rng, 300, 150, 0.5, n_detectors=4, shared=0.7)

    result = delong_auc_covariance(scores, labels, names=("a", "b", "c", "d"))

    np.testing.assert_allclose(result.covariance, result.covariance.T, atol=1e-15)
    assert np.linalg.eigvalsh(result.covariance).min() > -1e-14


def test_delong_contrast_matches_direct_variance_formula():
    rng = np.random.default_rng(41)
    scores, labels = _synthetic_auc_scores(rng, 120, 80, 0.7, n_detectors=3, shared=0.6)
    result = delong_auc_covariance(scores, labels, names=("p", "m", "c"))

    weights = {"p": 0.5, "m": 0.5, "c": -1.0}
    estimate, se = result.contrast(weights)

    c = np.array([0.5, 0.5, -1.0])
    assert estimate == pytest.approx(float(c @ result.aucs))
    assert se**2 == pytest.approx(float(c @ result.covariance @ c))


def test_delong_rejects_degenerate_label_groups():
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(2, 10))
    labels = np.ones(10, dtype=bool)
    with pytest.raises(ValueError, match="at least two items"):
        delong_auc_covariance(scores, labels, names=("a", "b"))


# --------------------------------------------------------------------------
# C4
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gap_baseline", "gap_target", "expected"),
    [
        (0.10, -0.10, True),  # forward reversal
        (-0.10, 0.10, True),  # reverse reversal
        (0.10, 0.05, False),  # gap shrinks but never crosses zero
        (-0.10, -0.05, False),  # both negative
    ],
)
def test_reversal_rule_over_all_four_sign_combinations(gap_baseline, gap_target, expected):
    assert is_reversal(gap_baseline, gap_target) is expected


@pytest.mark.parametrize(("gap_baseline", "gap_target"), [(0.0, -0.1), (0.1, 0.0), (0.0, 0.0)])
def test_zero_gap_is_not_a_reversal(gap_baseline, gap_target):
    assert is_reversal(gap_baseline, gap_target) is False


def test_c4_p_value_matches_the_intersection_union_definition():
    from scipy import stats

    gap_b, se_b, gap_q, se_q = 0.09, 0.02, -0.08, 0.02
    out = c4_p_value(gap_b, se_b, gap_q, se_q)

    p_plus_b = float(stats.norm.sf(gap_b / se_b))
    p_minus_b = float(stats.norm.cdf(gap_b / se_b))
    p_plus_q = float(stats.norm.sf(gap_q / se_q))
    p_minus_q = float(stats.norm.cdf(gap_q / se_q))

    assert out["p_forward"] == pytest.approx(max(p_plus_b, p_minus_q))
    assert out["p_reverse"] == pytest.approx(max(p_minus_b, p_plus_q))
    assert out["p_value"] == pytest.approx(
        min(1.0, 2 * min(out["p_forward"], out["p_reverse"]))
    )
    assert out["p_value"] < 0.05


def test_c4_p_value_is_capped_at_one_and_large_without_a_reversal():
    out = c4_p_value(0.09, 0.02, 0.08, 0.02)
    assert out["p_value"] == 1.0


def test_c4_gap_is_mean_of_probability_aucs_minus_cdd():
    """§4.5.6: g_p is an equal-weight mean of two AUCs, not an AUC of pooled
    raw scores, and both component AUCs are reported."""
    rng = np.random.default_rng(101)
    n_pos, n_neg = 690, 182
    labels = np.concatenate([np.ones(n_pos, dtype=bool), np.zeros(n_neg, dtype=bool)])

    def scores(sep):
        return rng.normal(size=n_pos + n_neg) + sep * labels

    baseline = {"perplexity": scores(0.9), "mink_prob": scores(0.85), "cdd": scores(0.1)}
    target = {"perplexity": scores(0.05), "mink_prob": scores(0.05), "cdd": scores(0.9)}

    result = c4_rank_reversal_test(baseline, target, labels)

    assert result.status == STATUS_COMPUTED
    assert result.n_possible_exposure == n_pos
    assert result.n_shared_clean_control == n_neg
    expected_gap_b = (
        empirical_auc(baseline["perplexity"], labels) + empirical_auc(baseline["mink_prob"], labels)
    ) / 2 - empirical_auc(baseline["cdd"], labels)
    assert result.gap_baseline == pytest.approx(expected_gap_b)
    assert set(result.component_aucs) == {
        "perplexity@bf16", "mink_prob@bf16", "cdd@bf16",
        "perplexity@bnb_nf4", "mink_prob@bnb_nf4", "cdd@bnb_nf4",
    }
    assert result.reversal_observed is True
    assert result.p_value < 0.05


def test_c4_no_reversal_gives_a_large_p_value():
    rng = np.random.default_rng(202)
    n_pos, n_neg = 400, 150
    labels = np.concatenate([np.ones(n_pos, dtype=bool), np.zeros(n_neg, dtype=bool)])

    def scores(sep):
        return rng.normal(size=n_pos + n_neg) + sep * labels

    baseline = {"perplexity": scores(0.9), "mink_prob": scores(0.9), "cdd": scores(0.1)}
    target = {"perplexity": scores(0.8), "mink_prob": scores(0.8), "cdd": scores(0.1)}

    result = c4_rank_reversal_test(baseline, target, labels)

    assert result.reversal_observed is False
    assert result.p_value > 0.5


def test_c4_constant_cdd_alone_does_not_make_it_not_estimable():
    """§4.5.6: "A constant CDD score alone is not a reason to remove C4 when
    both gap variances remain positive"."""
    rng = np.random.default_rng(303)
    n_pos, n_neg = 300, 120
    labels = np.concatenate([np.ones(n_pos, dtype=bool), np.zeros(n_neg, dtype=bool)])
    constant = np.full(n_pos + n_neg, 0.01)

    def scores(sep):
        return rng.normal(size=n_pos + n_neg) + sep * labels

    baseline = {"perplexity": scores(0.8), "mink_prob": scores(0.8), "cdd": constant}
    target = {"perplexity": scores(-0.8), "mink_prob": scores(-0.8), "cdd": constant}

    result = c4_rank_reversal_test(baseline, target, labels)

    assert result.status == STATUS_COMPUTED
    assert result.component_aucs["cdd@bf16"] == pytest.approx(0.5)
    assert result.se_gap_baseline > 0 and result.se_gap_target > 0


def test_c4_missing_detector_and_empty_label_group_are_not_estimable_with_p_one():
    rng = np.random.default_rng(404)
    labels = np.concatenate([np.ones(50, dtype=bool), np.zeros(30, dtype=bool)])
    full = {name: rng.normal(size=80) for name in ("perplexity", "mink_prob", "cdd")}

    missing = c4_rank_reversal_test({"perplexity": full["perplexity"]}, full, labels)
    assert missing.status == STATUS_MISSING_SCORES
    assert missing.p_value == 1.0
    assert missing.reversal_observed is False

    one_group = np.ones(80, dtype=bool)
    empty = c4_rank_reversal_test(full, full, one_group)
    assert empty.status == STATUS_EMPTY_LABEL_GROUP
    assert empty.p_value == 1.0


def test_c4_identical_precisions_give_zero_gap_variance_and_p_one():
    rng = np.random.default_rng(505)
    labels = np.concatenate([np.ones(100, dtype=bool), np.zeros(60, dtype=bool)])
    shared = {name: rng.normal(size=160) for name in ("perplexity", "mink_prob", "cdd")}
    # Make the probability family exactly the CDD scores so both gaps are 0.
    degenerate = {name: shared["cdd"] for name in ("perplexity", "mink_prob", "cdd")}

    result = c4_rank_reversal_test(degenerate, degenerate, labels)

    assert result.status == STATUS_ZERO_VARIANCE
    assert result.p_value == 1.0
    assert result.gap_baseline == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Holm
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p_values",
    [
        [0.001, 0.02, 0.03, 0.9],
        [1.0, 1.0, 1.0, 1.0],
        [0.01, 0.01, 0.01, 0.01],
        [0.04, 0.005, 0.5, 0.012],
        [0.0, 0.3, 0.049, 1.0],
        [0.2, 0.19, 0.18, 0.17, 0.16],
    ],
)
def test_holm_matches_statsmodels(p_values):
    from statsmodels.stats.multitest import multipletests

    reject, adjusted, _, _ = multipletests(p_values, alpha=0.05, method="holm")

    np.testing.assert_allclose(holm_adjusted_p_values(p_values), adjusted, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(holm_decisions(p_values, alpha=0.05), reject)


def test_holm_rejects_p_values_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_adjusted_p_values([0.1, 1.2])


# --------------------------------------------------------------------------
# The four-slot family
# --------------------------------------------------------------------------


def _q1a_inputs(rng, shifts):
    out = {}
    for detector, shift in shifts.items():
        before = rng.normal(size=1055)
        out[detector] = (before, before + rng.normal(shift, 1.0, 1055))
    return out


def test_family_keeps_four_slots_and_applies_holm():
    rng = np.random.default_rng(909)
    n_pos, n_neg = 690, 182
    labels = np.concatenate([np.ones(n_pos, dtype=bool), np.zeros(n_neg, dtype=bool)])

    def scores(sep):
        return rng.normal(size=n_pos + n_neg) + sep * labels

    family = run_confirmatory_family(
        _q1a_inputs(rng, {"perplexity": 0.25, "mink_prob": 0.0, "cdd": 0.4}),
        {"perplexity": scores(0.8), "mink_prob": scores(0.8), "cdd": scores(0.0)},
        {"perplexity": scores(0.0), "mink_prob": scores(0.0), "cdd": scores(0.8)},
        labels,
        model="Qwen2.5-32B-Instruct",
    )

    assert family.slots == CONFIRMATORY_SLOTS
    assert set(family.tests) == set(CONFIRMATORY_SLOTS)
    assert family.tests["C1"]["detector"] == "perplexity"
    assert family.tests["C2"]["detector"] == "mink_prob"
    assert family.tests["C3"]["detector"] == "cdd"
    assert family.tests["C1"]["n_pairs"] == 1055  # all LCB items, no exposure label
    np.testing.assert_allclose(
        [family.holm_adjusted_p_values[s] for s in CONFIRMATORY_SLOTS],
        holm_adjusted_p_values([family.raw_p_values[s] for s in CONFIRMATORY_SLOTS]),
    )
    assert family.rejected["C1"] is True
    assert family.rejected["C3"] is True


def test_family_retains_a_not_estimable_slot_with_p_one():
    """§4.5.6: "retain the slot with p=1 for multiplicity accounting rather
    than removing it or selecting another test"."""
    rng = np.random.default_rng(1010)
    before = rng.normal(size=100)
    q1a = {
        "perplexity": (before, before + rng.normal(0.5, 1.0, 100)),
        "mink_prob": (before, before.copy()),  # detector did not move -> zero variance
        "cdd": (before, before + rng.normal(0.0, 1.0, 100)),
    }
    labels = np.concatenate([np.ones(60, dtype=bool), np.zeros(40, dtype=bool)])
    scores = {name: rng.normal(size=100) for name in ("perplexity", "mink_prob", "cdd")}

    family = run_confirmatory_family(
        q1a, scores, {"perplexity": scores["perplexity"]}, labels, model="Qwen2.5-32B-Instruct"
    )

    assert len(family.raw_p_values) == 4
    assert family.tests["C2"]["status"] == STATUS_ZERO_VARIANCE
    assert family.raw_p_values["C2"] == 1.0
    assert family.tests["C4"]["status"] == STATUS_MISSING_SCORES
    assert family.raw_p_values["C4"] == 1.0
    # A p=1 slot still occupies a Holm step: the smallest raw p is multiplied
    # by 4, not by 2.
    assert family.holm_adjusted_p_values["C1"] == pytest.approx(
        min(1.0, 4 * family.raw_p_values["C1"])
    )


def test_family_rejects_a_missing_q1a_slot():
    empty_c4 = C4Result(
        status=STATUS_MISSING_SCORES,
        p_value=1.0,
        gap_baseline=None,
        gap_target=None,
        se_gap_baseline=None,
        se_gap_target=None,
        p_forward=None,
        p_reverse=None,
        reversal_observed=False,
        component_aucs={},
        n_possible_exposure=0,
        n_shared_clean_control=0,
    )
    with pytest.raises(ValueError, match="missing Q1a slots"):
        assemble_confirmatory_family({}, empty_c4, model="Qwen2.5-32B-Instruct")
