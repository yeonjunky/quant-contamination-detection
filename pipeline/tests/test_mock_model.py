"""Regression tests for MockModel's register-before-use contract (found and
fixed during pipeline construction: the original `LoadedModel` Protocol took
a `contaminated` kwarg on `generate()`/`score_logprobs()`, which would leak
the ground-truth contamination label into a real backend's generation
input — a real model must never see that. Fixed by moving ground truth to
`register_item()`, keyed by item_id, kept off the shared Protocol.)
"""

import pytest

from qcd.models.mock import MockModel


def test_generate_without_registration_raises():
    model = MockModel()
    with pytest.raises(KeyError, match="never register_item"):
        model.generate("unregistered", "prompt", temperature=0.0, sample_id=0)


def test_score_logprobs_without_registration_raises():
    model = MockModel()
    with pytest.raises(KeyError, match="never register_item"):
        model.score_logprobs("unregistered", [1, 2, 3])


def test_score_prompt_logprobs_uses_fixed_prompt_tokens():
    model = MockModel()
    model.register_item("x", contaminated=True, quality=1.0)
    prompt = "Solve this fixed problem"
    scores = model.score_prompt_logprobs("x", prompt)
    assert len(scores) == len(model.tokenizer.encode(prompt))
    assert all(score < 0 for score in scores)


def test_partial_pass_rate_without_registration_raises():
    model = MockModel()
    with pytest.raises(KeyError, match="never register_item"):
        model.partial_pass_rate("unregistered")


def test_contaminated_item_more_confident_than_clean():
    model = MockModel()
    model.register_item("hot", contaminated=True, quality=0.9)
    model.register_item("cold", contaminated=False, quality=0.9)

    hot = model.generate("hot", "prompt", temperature=0.8, sample_id=0)
    cold = model.generate("cold", "prompt", temperature=0.8, sample_id=0)

    import numpy as np

    assert np.mean(hot.token_logprobs) > np.mean(cold.token_logprobs)


def test_contaminated_item_repeats_across_samples_clean_item_does_not():
    model = MockModel()
    model.register_item("hot", contaminated=True, quality=0.9)
    model.register_item("cold", contaminated=False, quality=0.9)

    hot_0 = model.generate("hot", "prompt", temperature=0.8, sample_id=0)
    hot_1 = model.generate("hot", "prompt", temperature=0.8, sample_id=1)
    cold_0 = model.generate("cold", "prompt", temperature=0.8, sample_id=0)
    cold_1 = model.generate("cold", "prompt", temperature=0.8, sample_id=1)

    assert hot_0.token_ids == hot_1.token_ids
    assert cold_0.token_ids != cold_1.token_ids


def test_generate_signature_matches_loaded_model_protocol():
    # Guards against the original bug regressing: generate()/score_logprobs()
    # must not accept `contaminated` as a call-time kwarg.
    import inspect

    gen_params = set(inspect.signature(MockModel.generate).parameters)
    score_params = set(inspect.signature(MockModel.score_logprobs).parameters)
    assert "contaminated" not in gen_params
    assert "contaminated" not in score_params
