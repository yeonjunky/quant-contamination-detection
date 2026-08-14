"""Regression tests against paper/paper_draft.md §4.5.3's worked table —
verified against the draft during pipeline construction (and independently
re-verified again during the round-8 review's own verify_stats.py pass, see
review/review_findings_round8.md).
"""

import pytest

from qcd.analysis.logodds import (
    aggregate_log_odds_bias,
    implied_correlation_at_p50,
    marginal_accuracy,
    solve_intercept_for_base_rate,
    spurious_pp_interaction,
)
from qcd.constants import BASE_RATE_HUMANEVAL_ILLUSTRATIVE, BASE_RATE_LCB_POST_ILLUSTRATIVE, IMPLIED_R_AT_P50

_KW = dict(humaneval_base_rate=BASE_RATE_HUMANEVAL_ILLUSTRATIVE, lcb_post_base_rate=BASE_RATE_LCB_POST_ILLUSTRATIVE)


def test_solve_intercept_recovers_target_base_rate():
    alpha = solve_intercept_for_base_rate(0.85)
    assert marginal_accuracy(alpha) == pytest.approx(0.85, abs=1e-4)


@pytest.mark.parametrize(
    "beta,he_drop,lcb_drop,interaction",
    [
        (0.25, 2.6, 4.0, -1.4),
        (0.50, 5.6, 7.7, -2.2),
        (0.75, 8.8, 11.2, -2.5),
        (1.00, 12.3, 14.5, -2.2),
    ],
)
def test_spurious_pp_interaction_table(beta, he_drop, lcb_drop, interaction):
    row = spurious_pp_interaction(beta, **_KW)
    assert row["humaneval_drop_pp"] == pytest.approx(he_drop, abs=0.05)
    assert row["lcb_post_drop_pp"] == pytest.approx(lcb_drop, abs=0.05)
    assert row["spurious_interaction_pp"] == pytest.approx(interaction, abs=0.05)


def test_implied_correlation_matches_constant():
    assert implied_correlation_at_p50() == pytest.approx(IMPLIED_R_AT_P50, abs=1e-5)


def test_aggregate_log_odds_bias_max_around_beta_075():
    # Paper: "a systematic bias of up to about 0.023" — max over the worked
    # table's beta range, at beta=0.75.
    bias_at_075 = aggregate_log_odds_bias(0.75, **_KW)
    assert bias_at_075 == pytest.approx(0.023, abs=0.001)

    for beta in (0.25, 0.50, 1.00):
        assert abs(aggregate_log_odds_bias(beta, **_KW)) <= abs(bias_at_075) + 1e-6
