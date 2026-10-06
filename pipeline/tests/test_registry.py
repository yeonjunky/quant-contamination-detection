from qcd.models.registry import ALL_MODELS, OLMO3_1_32B, get_model


def test_every_model_is_pinned_to_an_immutable_commit() -> None:
    assert len(ALL_MODELS) == 5
    assert all(len(model.revision) == 40 for model in ALL_MODELS)
    assert all(set(model.revision) <= set("0123456789abcdef") for model in ALL_MODELS)
    assert all(model.primary_first_post_boundary is not None for model in ALL_MODELS)


def test_olmo_3_1_32b_uses_verified_hugging_face_id() -> None:
    assert OLMO3_1_32B.name == "Olmo3.1-32B-Instruct"
    assert OLMO3_1_32B.hf_repo_id == "allenai/Olmo-3.1-32B-Instruct"
    assert get_model("Olmo3.1-32B-Instruct") is OLMO3_1_32B
    assert OLMO3_1_32B in ALL_MODELS


def test_invalid_olmo_3_32b_id_is_not_in_roster() -> None:
    assert all(model.hf_repo_id != "allenai/Olmo-3-32B-Instruct" for model in ALL_MODELS)


def test_sample_batch_size_is_fixed_for_small_models_and_unmeasured_for_32b() -> None:
    assert {model.name: model.sample_batch_size for model in ALL_MODELS} == {
        "Qwen2.5-7B-Instruct": 50,
        "Llama-3.1-8B-Instruct": 50,
        "Olmo3-7B-Instruct": 50,
        "Qwen2.5-32B-Instruct": None,
        "Olmo3.1-32B-Instruct": None,
    }
