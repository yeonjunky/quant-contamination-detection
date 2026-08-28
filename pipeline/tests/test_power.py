"""Regression tests against paper §4.5's worked power/sample-size tables —
values independently re-verified during the round-8 review (see
review/review_findings_round8.md)."""

import pytest

from qcd.analysis.power import (
    items_needed_diff_in_diff,
    items_needed_paired_t,
    items_needed_with_label_noise,
    minimum_detectable_diff_in_diff_unequal,
    power_diff_in_diff,
)


def test_items_needed_paired_t_matches_paper():
    assert items_needed_paired_t(0.3) == 87
    assert items_needed_paired_t(0.2) == 196


@pytest.mark.parametrize(
    "n,delta_pp,expected_power",
    [
        (50, 5, 0.06), (50, 10, 0.11), (50, 20, 0.29),
        (164, 5, 0.10), (164, 10, 0.25), (164, 20, 0.73),
        (400, 5, 0.17), (400, 10, 0.52), (400, 20, 0.98),
        (800, 5, 0.29), (800, 10, 0.81), (800, 20, 1.00),
        (1600, 5, 0.52), (1600, 10, 0.98), (1600, 20, 1.00),
        (3200, 5, 0.81), (3200, 10, 1.00), (3200, 20, 1.00),
    ],
)
def test_power_diff_in_diff_grid_matches_paper(n, delta_pp, expected_power):
    assert power_diff_in_diff(n, delta_pp) == pytest.approx(expected_power, abs=0.005)


def test_items_needed_diff_in_diff_matches_paper():
    # A ±1 slop is a pre-existing, already-documented rounding-convention
    # ambiguity (review/review_findings_round7.md: "170/171... 경계 관례 차이"),
    # not a bug — the paper's displayed integer is a nearest-value rounding
    # of a continuous bisection root, this code takes a strict ceiling.
    assert items_needed_diff_in_diff(20) == pytest.approx(196, abs=1)
    assert items_needed_diff_in_diff(10) == pytest.approx(785, abs=1)
    assert items_needed_diff_in_diff(5) == pytest.approx(3140, abs=1)


def test_primary_q2_lcb_ceiling_mde_matches_revised_design():
    assert minimum_detectable_diff_in_diff_unequal(float("inf"), 182) == pytest.approx(14.684, abs=0.001)
    assert minimum_detectable_diff_in_diff_unequal(873, 182) == pytest.approx(16.143, abs=0.001)
    assert minimum_detectable_diff_in_diff_unequal(164, float("inf")) == pytest.approx(15.469, abs=0.001)


def test_unequal_auc_se_reduces_to_equal_group_formula():
    from qcd.analysis.auc import hanley_mcneil_se, hanley_mcneil_se_unequal

    assert hanley_mcneil_se_unequal(0.70, 164, 164) == pytest.approx(
        hanley_mcneil_se(0.70, 164)
    )


def test_items_needed_with_label_noise_matches_paper():
    # true_delta_auc=0.050, true_auc=0.70, r=0.8 -> 170/287/541/1268 at
    # e=0/.1/.2/.3 (CLAUDE.md §4.1's adopted "SE도 감쇠" column). Same ±1
    # rounding-convention slop as items_needed_diff_in_diff above.
    assert items_needed_with_label_noise(0.050, 0.0, true_auc=0.70, r=0.8) == pytest.approx(170, abs=1)
    assert items_needed_with_label_noise(0.050, 0.10, true_auc=0.70, r=0.8) == pytest.approx(287, abs=1)
    assert items_needed_with_label_noise(0.050, 0.20, true_auc=0.70, r=0.8) == pytest.approx(541, abs=1)
    assert items_needed_with_label_noise(0.050, 0.30, true_auc=0.70, r=0.8) == 1268


def test_items_needed_with_label_noise_requires_se_attenuation_not_just_delta():
    # Regression guard for the found-and-fixed bug: passing the unattenuated
    # true_auc as the SE baseline (the old, wrong behavior) reproduces the
    # non-adopted "SE 고정" column (266/472/1060) instead of the adopted
    # "SE도 감쇠" column (287/541/1268) — these must NOT match.
    from qcd.analysis.auc import items_needed_for_delta_auc

    se_fixed_wrong = items_needed_for_delta_auc(0.8 * 0.050, 0.70, 0.8)  # e=0.10 -> (1-2e)=0.8, delta attenuated only
    se_also_attenuated_correct = items_needed_with_label_noise(0.050, 0.10, true_auc=0.70, r=0.8)
    assert se_fixed_wrong != se_also_attenuated_correct
    assert se_fixed_wrong == pytest.approx(266, abs=1)
    assert se_also_attenuated_correct == pytest.approx(287, abs=1)
