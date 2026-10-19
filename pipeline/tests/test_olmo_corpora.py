"""E-F14: the two Olmo arms do not share a training corpus.

`OLMO_GROUND_TRUTH.md`'s verified-inputs table records different bulk
pretraining mixes for `Olmo3-7B-Instruct` and `Olmo3.1-32B-Instruct`, and
paper §4.2 stores corpus evidence per model. These tests pin the separation and
the places where the registry refuses to guess.
"""

import pytest

from qcd.data.schema import CorpusReferenceStatus
from qcd.ground_truth.olmo_corpora import (
    CLOSED_CORPUS_MODELS, OLMO_CORPORA, OPEN_CORPUS_MODELS, check_model_corpus,
    corpora_for_model, default_status, find_corpus, require_revision, unassigned_corpora,
)
from qcd.models.registry import MAIN_ANALYSIS_MODELS


def test_registry_model_names_match_models_registry():
    known = {model.name for model in MAIN_ANALYSIS_MODELS}
    assert set(OPEN_CORPUS_MODELS) | set(CLOSED_CORPUS_MODELS) == known


def test_the_two_olmo_arms_have_different_pretraining_mixes():
    seven = [c for c in corpora_for_model("Olmo3-7B-Instruct") if c.stage == "pretraining"]
    thirty_two = [c for c in corpora_for_model("Olmo3.1-32B-Instruct") if c.stage == "pretraining"]
    assert [c.hf_repo for c in seven] == ["allenai/dolma3_mix-6T-1025-7B"]
    assert [c.hf_repo for c in thirty_two] == ["allenai/dolma3_mix-6T"]


def test_every_open_corpus_is_pinned_to_a_hub_revision():
    pins = {corpus.hf_repo: require_revision(corpus) for corpus in OLMO_CORPORA}
    assert pins == {
        "allenai/dolma3_mix-6T-1025-7B": "2ca900fbe14e86c5c83d064d9f0882f1c0b8c05b",
        "allenai/dolma3_mix-6T": "689a3ea2d8217e64d73a5058913fa43ad15e81aa",
        "allenai/Dolci-Instruct-SFT": "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221",
        "allenai/Dolci-Instruct-DPO": "aed155cf32e809b590490b6c3577ee4b0d0a5019",
        "allenai/Dolci-Instruct-RL": "1a8d79b1012d77e4bb1fd7f699c5c7eb33ef931e",
    }


def test_an_unpinned_corpus_is_refused():
    import dataclasses

    unpinned = dataclasses.replace(find_corpus("allenai/dolma3_mix-6T"), revision=None)
    with pytest.raises(ValueError, match="no pinned scan revision"):
        require_revision(unpinned)


def test_post_training_repositories_belong_to_both_instruct_arms():
    """Each arm's model card names the same three Dolci-Instruct repositories."""
    assert unassigned_corpora() == ()
    for model in OPEN_CORPUS_MODELS:
        stages = {c.stage: c.hf_repo for c in corpora_for_model(model) if c.stage != "pretraining"}
        assert stages == {
            "sft": "allenai/Dolci-Instruct-SFT",
            "dpo": "allenai/Dolci-Instruct-DPO",
            "rlvr": "allenai/Dolci-Instruct-RL",
        }
        assert check_model_corpus(model, "allenai/Dolci-Instruct-SFT") is None


def test_scanning_the_other_arms_pretraining_mix_is_refused():
    with pytest.raises(ValueError, match="not Olmo3.1-32B-Instruct"):
        check_model_corpus("Olmo3.1-32B-Instruct", "allenai/dolma3_mix-6T-1025-7B")
    assert check_model_corpus("Olmo3-7B-Instruct", "allenai/dolma3_mix-6T-1025-7B") is None


def test_unknown_pairings_warn_instead_of_passing_silently():
    assert "unverified" in check_model_corpus("Olmo3-7B-Instruct", "someone/unknown-mix")


def test_closed_corpus_arms_are_not_observable_and_have_no_scan():
    for model in CLOSED_CORPUS_MODELS:
        assert default_status(model) is CorpusReferenceStatus.NOT_OBSERVABLE
        assert corpora_for_model(model) == ()
        with pytest.raises(ValueError, match="no released training corpus"):
            check_model_corpus(model, "allenai/dolma3_mix-6T")
    for model in OPEN_CORPUS_MODELS:
        assert default_status(model) is None
