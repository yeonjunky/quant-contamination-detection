import pytest

from qcd.pilot.pilot_report import (
    ValidationDiagnostics,
    base_rate,
    cohens_d_paired,
    pearson_r,
)


# --- validation-diagnostic primitives ---------------------------------------


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


def test_validation_diagnostics_default_construction():
    report = ValidationDiagnostics()
    assert report.q1a_effect_size_d == {}
