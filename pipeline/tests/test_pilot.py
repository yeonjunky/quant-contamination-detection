import pytest

from qcd.constants import CDD_GATE_AUC
from qcd.pilot.cdd_gate import check_cdd_gate
from qcd.pilot.pilot_report import PilotReport, base_rate, cohens_d_paired, pearson_r, proxy_label_error_rate


# --- CDD gate ----------------------------------------------------------------


def test_cdd_gate_passes_above_threshold():
    result = check_cdd_gate(CDD_GATE_AUC + 0.05)
    assert result.passed is True
    assert "stays in the primary analysis" in result.reason


def test_cdd_gate_fails_below_threshold():
    result = check_cdd_gate(CDD_GATE_AUC - 0.05)
    assert result.passed is False
    assert "drop CDD" in result.reason


def test_cdd_gate_boundary_at_exact_threshold_passes():
    result = check_cdd_gate(CDD_GATE_AUC)
    assert result.passed is True


def test_cdd_gate_reports_diagnostics():
    result = check_cdd_gate(0.85)
    assert result.assumed_quantization_delta_auc > 0
    assert result.detection_limit_at_reference_n > 0


# --- pilot_report primitives -------------------------------------------------


def test_cohens_d_paired_known_answer():
    before = [1.0, 2.0, 3.0, 4.0]
    after = [2.0, 3.0, 4.0, 5.0]  # constant +1 shift -> SD(diff)=0 -> d=0.0 (guarded)
    assert cohens_d_paired(before, after) == 0.0


def test_cohens_d_paired_nonzero_variance():
    before = [1.0, 2.0, 3.0, 4.0, 5.0]
    after = [1.5, 2.0, 4.0, 4.5, 7.0]
    d = cohens_d_paired(before, after)
    assert d > 0  # after > before on average


def test_cohens_d_paired_needs_two_points():
    with pytest.raises(ValueError):
        cohens_d_paired([1.0], [2.0])


def test_pearson_r_perfect_correlation():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    assert pearson_r(x, y) == pytest.approx(1.0)


def test_pearson_r_no_correlation_when_constant():
    assert pearson_r([1, 1, 1], [1, 2, 3]) == 0.0


def test_base_rate_known_answer():
    assert base_rate([1, 1, 0, 0]) == pytest.approx(0.5)
    assert base_rate([True, True, True, False]) == pytest.approx(0.75)


def test_base_rate_empty_raises():
    with pytest.raises(ValueError):
        base_rate([])


def test_proxy_label_error_rate_known_answer():
    proxy = [True, True, False, False]
    truth = [True, False, False, False]
    # one disagreement (index 1) out of 4 -> e = 0.25
    assert proxy_label_error_rate(proxy, truth) == pytest.approx(0.25)


def test_proxy_label_error_rate_perfect_agreement_is_zero():
    labels = [True, False, True, False]
    assert proxy_label_error_rate(labels, labels) == 0.0


def test_pilot_report_default_construction():
    report = PilotReport()
    assert report.q1a_effect_size_d == {}
    assert report.olmo3_proxy_label_error_rate is None
