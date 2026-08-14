import math

import pytest

from qcd.constants import CDD_EDIT_DISTANCE_ALPHA, CDD_MAX_TOKENS, CDD_XI_FIXED
from qcd.detectors.cdd import peakedness, token_edit_distance
from qcd.detectors.mink_prob import mink_prob
from qcd.detectors.perplexity import negative_log_perplexity_score, perplexity
from qcd.detectors.threshold import (
    ThresholdSelectionOnEvalSetError,
    classify,
    select_threshold_youden,
    youden_j,
)

# --- CDD ---------------------------------------------------------------


def test_token_edit_distance_identical_is_zero():
    assert token_edit_distance([1, 2, 3], [1, 2, 3]) == 0


def test_token_edit_distance_known_values():
    assert token_edit_distance([], [1, 2, 3]) == 3
    assert token_edit_distance([1, 2, 3], []) == 3
    assert token_edit_distance([1, 2, 3], [1, 2, 4]) == 1  # one substitution
    assert token_edit_distance([1, 2, 3], [1, 2]) == 1  # one deletion


def test_peakedness_identical_streams_is_one():
    greedy = list(range(100))
    samples = [list(range(100)) for _ in range(10)]
    assert peakedness(greedy, samples) == 1.0


def test_peakedness_maximally_different_streams_is_zero():
    greedy = [0] * CDD_MAX_TOKENS
    # every position substituted -> edit distance = CDD_MAX_TOKENS, far above
    # the alpha*max_tokens=5 threshold.
    samples = [[1] * CDD_MAX_TOKENS for _ in range(10)]
    assert peakedness(greedy, samples) == 0.0


def test_peakedness_threshold_boundary():
    threshold = int(CDD_EDIT_DISTANCE_ALPHA * CDD_MAX_TOKENS)  # 5
    greedy = [0] * CDD_MAX_TOKENS
    exactly_at_threshold = [0] * (CDD_MAX_TOKENS - threshold) + [1] * threshold
    one_over_threshold = [0] * (CDD_MAX_TOKENS - threshold - 1) + [1] * (threshold + 1)

    assert peakedness(greedy, [exactly_at_threshold]) == 1.0
    assert peakedness(greedy, [one_over_threshold]) == 0.0


def test_peakedness_truncates_to_max_tokens():
    # Sequences differ only beyond CDD_MAX_TOKENS -> truncation makes them
    # identical for scoring purposes.
    greedy = [0] * CDD_MAX_TOKENS + [9] * 50
    sample = [0] * CDD_MAX_TOKENS + [7] * 50
    assert peakedness(greedy, [sample]) == 1.0


def test_peakedness_empty_samples_raises():
    with pytest.raises(ValueError):
        peakedness([1, 2, 3], [])


# --- Perplexity ----------------------------------------------------------


def test_perplexity_uniform_confidence():
    # log(p) = log(0.5) for every token -> perplexity = 1/0.5 = 2
    logprobs = [math.log(0.5)] * 10
    assert perplexity(logprobs) == pytest.approx(2.0)


def test_perplexity_lower_for_more_confident_sequence():
    confident = [math.log(0.9)] * 10
    unconfident = [math.log(0.3)] * 10
    assert perplexity(confident) < perplexity(unconfident)


def test_negative_log_perplexity_score_higher_for_confident():
    confident = [math.log(0.9)] * 10
    unconfident = [math.log(0.3)] * 10
    assert negative_log_perplexity_score(confident) > negative_log_perplexity_score(unconfident)


def test_perplexity_empty_raises():
    with pytest.raises(ValueError):
        perplexity([])


# --- Min-k% Prob -----------------------------------------------------------


def test_mink_prob_known_answer():
    # 10 tokens, logprobs 0..-9 (i.e. -0, -1, ..., -9). k=20% of 10 = 2
    # lowest (most negative) log-probs: -9, -8 -> mean -8.5.
    logprobs = [-float(i) for i in range(10)]
    assert mink_prob(logprobs, k_percent=20.0) == pytest.approx(-8.5)


def test_mink_prob_k100_equals_mean():
    logprobs = [-1.0, -2.0, -3.0, -4.0]
    assert mink_prob(logprobs, k_percent=100.0) == pytest.approx(sum(logprobs) / len(logprobs))


def test_mink_prob_higher_for_confident_sequence():
    # A sequence with occasional very-low-probability outlier tokens should
    # score lower (more negative) under Min-k% than one without outliers,
    # matching the paper's own hypothesis ("unseen example is likely to
    # contain a few outlier words with low probabilities").
    with_outliers = [-0.1, -0.2, -0.15, -0.1, -8.0, -9.0]
    without_outliers = [-0.1, -0.2, -0.15, -0.1, -0.3, -0.25]
    assert mink_prob(with_outliers, k_percent=20.0) < mink_prob(without_outliers, k_percent=20.0)


def test_mink_prob_at_least_one_token_selected_for_short_sequences():
    # k_percent=1 on a 3-token sequence would round to 0 without the max(1, ...) floor.
    assert mink_prob([-1.0, -2.0, -3.0], k_percent=1.0) == pytest.approx(-3.0)


def test_mink_prob_invalid_k_percent_raises():
    with pytest.raises(ValueError):
        mink_prob([-1.0], k_percent=0.0)
    with pytest.raises(ValueError):
        mink_prob([-1.0], k_percent=101.0)


# --- Threshold ---------------------------------------------------------------


def test_classify_uses_fixed_xi_by_default():
    scores = [0.0, CDD_XI_FIXED, CDD_XI_FIXED + 0.001]
    assert classify(scores) == [False, False, True]


def test_youden_j_perfect_separation():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [False, False, True, True]
    assert youden_j(scores, labels, threshold=0.5) == pytest.approx(1.0)


def test_select_threshold_youden_recovers_perfect_separator():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [False, False, True, True]
    threshold, j = select_threshold_youden(scores, labels)
    assert j == pytest.approx(1.0)
    assert 0.2 <= threshold < 0.8


def test_select_threshold_youden_raises_on_eval_set_overlap():
    scores = [0.1, 0.9]
    labels = [False, True]
    with pytest.raises(ThresholdSelectionOnEvalSetError):
        select_threshold_youden(
            scores, labels,
            calibration_item_ids=["a", "b"],
            evaluation_item_ids=["b", "c"],
        )


def test_select_threshold_youden_allows_disjoint_sets():
    scores = [0.1, 0.9]
    labels = [False, True]
    # Should not raise.
    select_threshold_youden(
        scores, labels,
        calibration_item_ids=["a", "b"],
        evaluation_item_ids=["c", "d"],
    )
