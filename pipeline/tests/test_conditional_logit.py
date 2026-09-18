"""Item-stratified conditional logistic interval for β_QE — paper §4.5.5.

Cross-checks:

- the design matrix against §3.1's coding (Q=0 bf16, Q=1 quantized, E=0
  shared control, E=1 possible exposure; regressors Q and Q*E);
- the fitted coefficients against `statsmodels`' own `ConditionalLogit` run
  directly, so the wrapper is verified not to reorder or rescale anything;
- the two-observation-per-item case against the closed-form conditional
  (matched-pair) logistic likelihood, where β is the log of the ratio of
  discordant-pair counts;
- recovery of a known β_QE on design-shaped synthetic data;
- `J = -β_QE` with reversed interval endpoints.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from qcd.analysis.conditional_logit import (
    STATUS_COMPUTED,
    STATUS_NOT_ESTIMABLE,
    beta_qe_interval_to_j,
    build_qe_design,
    fit_conditional_logit_beta_qe,
    fit_conditional_logit_by_model,
)
from qcd.analysis.coverage import SyntheticDesign, simulate_design_dataset


def _frame(rows):
    return pd.DataFrame(rows, columns=["item_id", "precision", "exposure_proxy", "correct"])


def test_design_matrix_follows_section_3_1_coding():
    df = _frame(
        [
            ("a", "bf16", True, 1),
            ("a", "bnb_nf4", True, 0),
            ("b", "bf16", False, 0),
            ("b", "bnb_nf4", False, 1),
        ]
    )

    y, x, groups = build_qe_design(df, baseline="bf16", target="bnb_nf4")

    np.testing.assert_array_equal(y, [1, 0, 0, 1])
    np.testing.assert_array_equal(x[:, 0], [0.0, 1.0, 0.0, 1.0])  # Q
    np.testing.assert_array_equal(x[:, 1], [0.0, 1.0, 0.0, 0.0])  # Q*E
    np.testing.assert_array_equal(groups, ["a", "a", "b", "b"])


def test_design_matrix_drops_other_precisions():
    df = _frame(
        [
            ("a", "bf16", True, 1),
            ("a", "bnb_nf4", True, 0),
            ("a", "gptq_awq_int4", True, 1),
        ]
    )
    y, x, _ = build_qe_design(df, baseline="bf16", target="bnb_nf4")
    assert y.size == 2 and x.shape == (2, 2)


def test_exposure_must_be_constant_within_item():
    df = _frame(
        [
            ("a", "bf16", True, 1),
            ("a", "bnb_nf4", False, 0),
        ]
    )
    with pytest.raises(ValueError, match="constant across precision"):
        build_qe_design(df, baseline="bf16", target="bnb_nf4")


def test_matches_statsmodels_conditional_logit_directly():
    from statsmodels.discrete.conditional_models import ConditionalLogit

    design = SyntheticDesign(beta_qe=0.5)
    df = simulate_design_dataset(design, np.random.default_rng(12))

    result = fit_conditional_logit_beta_qe(df, baseline="bf16", target="bnb_nf4")
    y, x, groups = build_qe_design(df, baseline="bf16", target="bnb_nf4")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels announces dropped concordant strata
        reference = ConditionalLogit(y, x, groups=groups).fit(disp=0)

    assert result.status == STATUS_COMPUTED
    assert result.beta_q == pytest.approx(float(reference.params[0]), rel=1e-10)
    assert result.beta_qe == pytest.approx(float(reference.params[1]), rel=1e-10)
    assert result.standard_error == pytest.approx(float(reference.bse[1]), rel=1e-10)
    low, high = reference.conf_int()[1]
    assert result.ci_low == pytest.approx(float(low), rel=1e-8)
    assert result.ci_high == pytest.approx(float(high), rel=1e-8)


def test_matched_pair_closed_form_recovers_beta_q():
    """With two observations per item and E identically 0, the conditional
    likelihood is McNemar's matched-pair logistic: the MLE is
    ``β_Q = log(b / c)`` with b = (0 at bf16, 1 at nf4) and c the reverse."""
    b_count, c_count = 60, 20
    rows = []
    index = 0
    for _ in range(b_count):
        rows += [(f"i{index}", "bf16", False, 0), (f"i{index}", "bnb_nf4", False, 1)]
        index += 1
    for _ in range(c_count):
        rows += [(f"i{index}", "bf16", False, 1), (f"i{index}", "bnb_nf4", False, 0)]
        index += 1
    # Concordant items contribute nothing and must not change the estimate.
    for _ in range(50):
        rows += [(f"i{index}", "bf16", False, 1), (f"i{index}", "bnb_nf4", False, 1)]
        index += 1
    # At least one exposed discordant pair so Q*E is not constant.
    for outcome in ((0, 1), (1, 0)):
        rows += [
            (f"i{index}", "bf16", True, outcome[0]),
            (f"i{index}", "bnb_nf4", True, outcome[1]),
        ]
        index += 1

    result = fit_conditional_logit_beta_qe(_frame(rows))

    assert result.status == STATUS_COMPUTED
    assert result.beta_q == pytest.approx(np.log(b_count / c_count), abs=1e-3)
    # Q*E has one pair each way -> β_QE must offset β_Q back to zero.
    assert result.beta_q + result.beta_qe == pytest.approx(0.0, abs=1e-3)
    assert result.n_items_supplied == index
    assert result.n_items_informative == b_count + c_count + 2
    assert result.n_observations_used == 2 * (b_count + c_count + 2)


def test_recovers_a_known_beta_qe_on_design_shaped_data():
    design = SyntheticDesign(beta_qe=0.8)
    fits = [
        fit_conditional_logit_beta_qe(
            simulate_design_dataset(design, np.random.default_rng(seed))
        )
        for seed in range(12)
    ]
    estimates = np.array([f.beta_qe for f in fits])

    assert all(f.status == STATUS_COMPUTED for f in fits)
    assert estimates.mean() == pytest.approx(0.8, abs=0.25)
    assert sum(f.covers(0.8) for f in fits) >= 10  # 12 draws at nominal 95%


def test_j_is_the_negated_interval_with_reversed_endpoints():
    assert beta_qe_interval_to_j(-0.2, 1.1) == (-1.1, 0.2)
    assert beta_qe_interval_to_j(None, 1.1) is None

    design = SyntheticDesign(beta_qe=0.5)
    result = fit_conditional_logit_beta_qe(
        simulate_design_dataset(design, np.random.default_rng(99))
    )
    assert result.j == pytest.approx(-result.beta_qe)
    low, high = result.j_interval
    assert low == pytest.approx(-result.ci_high)
    assert high == pytest.approx(-result.ci_low)
    assert low < high
    payload = result.as_dict()
    assert payload["j_ci_low"] == pytest.approx(low)
    assert payload["j_ci_high"] == pytest.approx(high)
    assert payload["interval_method"] == "item_stratified_conditional_logit_wald"


def test_no_informative_item_is_not_estimable_rather_than_an_exception():
    rows = []
    for index in range(20):
        exposure = index < 10
        rows += [
            (f"i{index}", "bf16", exposure, 1),
            (f"i{index}", "bnb_nf4", exposure, 1),
        ]

    result = fit_conditional_logit_beta_qe(_frame(rows))

    assert result.status == STATUS_NOT_ESTIMABLE
    assert result.beta_qe is None and result.ci_low is None and result.ci_high is None
    assert result.covers(0.0) is None
    assert "within-item outcome difference" in result.message


def test_single_exposure_group_is_not_estimable():
    rows = []
    for index in range(20):
        rows += [
            (f"i{index}", "bf16", False, index % 2),
            (f"i{index}", "bnb_nf4", False, (index + 1) % 2),
        ]

    result = fit_conditional_logit_beta_qe(_frame(rows))

    assert result.status == STATUS_NOT_ESTIMABLE
    assert "not identified" in result.message


def test_per_model_fits_are_independent():
    """§4.5.5: the conditional fit is "run separately per model"."""
    design = SyntheticDesign(beta_qe=0.5)
    frames = []
    for name, seed in (("alpha", 3), ("beta", 4)):
        df = simulate_design_dataset(design, np.random.default_rng(seed))
        df = df.assign(model=name, item_id=df["item_id"].astype(str) + f"_{name}")
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)

    by_model = fit_conditional_logit_by_model(pooled)
    assert set(by_model) == {"alpha", "beta"}
    for name, frame in zip(("alpha", "beta"), frames):
        alone = fit_conditional_logit_beta_qe(frame)
        assert by_model[name].beta_qe == pytest.approx(alone.beta_qe, rel=1e-10)
        assert by_model[name].ci_low == pytest.approx(alone.ci_low, rel=1e-10)


def test_concordant_exposure_group_is_not_estimable():
    """Only discordant items enter the conditional likelihood, so if every
    `possible-exposure` item is concordant, β_QE is flat and the fit must be
    reported as not estimable rather than as a number."""
    rows = []
    index = 0
    for k in range(40):
        rows += [
            (f"i{index}", "bf16", False, k % 2),
            (f"i{index}", "bnb_nf4", False, (k + 1) % 2),
        ]
        index += 1
    for _ in range(20):
        rows += [(f"i{index}", "bf16", True, 1), (f"i{index}", "bnb_nf4", True, 1)]
        index += 1

    result = fit_conditional_logit_beta_qe(_frame(rows))

    assert result.status == STATUS_NOT_ESTIMABLE
    assert result.beta_qe is None
    assert "not identified" in result.message
    assert result.n_items_supplied == 60
    assert result.n_items_informative == 40
