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


def test_only_the_7b_pretraining_mix_has_a_pinned_scan_revision():
    """The 32B revision is left unresolved rather than invented."""
    seven = find_corpus("allenai/dolma3_mix-6T-1025-7B")
    thirty_two = find_corpus("allenai/dolma3_mix-6T")
    assert require_revision(seven) == "2ca900fbe14e86c5c83d064d9f0882f1c0b8c05b"
    assert thirty_two.revision is None
    with pytest.raises(ValueError, match="no pinned scan revision"):
        require_revision(thirty_two)


def test_post_training_repositories_are_not_attributed_to_a_checkpoint():
    unassigned = {corpus.hf_repo for corpus in unassigned_corpora()}
    assert unassigned == {
        "allenai/Dolci-Instruct-SFT",
        "allenai/Dolci-Instruct-DPO",
        "allenai/Dolci-Instruct-RL",
    }
    for corpus in OLMO_CORPORA:
        if corpus.hf_repo in unassigned:
            assert corpus.assignment_verified is False


def test_scanning_the_other_arms_pretraining_mix_is_refused():
    with pytest.raises(ValueError, match="not Olmo3.1-32B-Instruct"):
        check_model_corpus("Olmo3.1-32B-Instruct", "allenai/dolma3_mix-6T-1025-7B")
    assert check_model_corpus("Olmo3-7B-Instruct", "allenai/dolma3_mix-6T-1025-7B") is None


def test_unverified_or_unknown_pairings_warn_instead_of_passing_silently():
    assert "operator-declared" in check_model_corpus(
        "Olmo3-7B-Instruct", "allenai/Dolci-Instruct-SFT"
    )
    assert "unverified" in check_model_corpus("Olmo3-7B-Instruct", "someone/unknown-mix")


def test_closed_corpus_arms_are_not_observable_and_have_no_scan():
    for model in CLOSED_CORPUS_MODELS:
        assert default_status(model) is CorpusReferenceStatus.NOT_OBSERVABLE
        assert corpora_for_model(model) == ()
        with pytest.raises(ValueError, match="no released training corpus"):
            check_model_corpus(model, "allenai/dolma3_mix-6T")
    for model in OPEN_CORPUS_MODELS:
        assert default_status(model) is None
