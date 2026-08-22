import numpy as np
import pytest

from qcd.ground_truth.tracer_embedding import JinaEmbeddingAdapter, JinaEmbeddingConfig


REVISION = "ab036b023d30b4d1138c4c3bfa9f0c445ab455d6"


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        vectors = {"x": [3.0, 4.0], "y": [0.0, 2.0], "minus": [-3.0, -4.0]}
        return [vectors[text] for text in texts]


def test_config_requires_revision_and_records_reimplementation_choices():
    config = JinaEmbeddingConfig(revision=REVISION, max_length=2048, truncate_dim=256)
    assert config.manifest_record() == {
        "model_id": "jinaai/jina-embeddings-v3",
        "revision": REVISION,
        "task": "text-matching",
        "max_length": 2048,
        "truncate_dim": 256,
        "similarity": "cosine_clipped_0_1",
    }


@pytest.mark.parametrize("revision", ["", "main", "A" * 40, "a" * 39])
def test_config_rejects_mutable_or_malformed_revision(revision):
    with pytest.raises(ValueError):
        JinaEmbeddingConfig(revision=revision)


def test_adapter_is_lazy_and_passes_all_frozen_encode_choices():
    fake = FakeModel()
    loads = []
    adapter = JinaEmbeddingAdapter(
        JinaEmbeddingConfig(revision=REVISION, max_length=1024, truncate_dim=2),
        model_loader=lambda config: loads.append(config) or fake,
    )
    assert loads == []
    vectors = adapter.encode(["x", "y"])
    assert len(loads) == 1
    assert fake.calls == [
        (["x", "y"], {"task": "text-matching", "max_length": 1024, "truncate_dim": 2})
    ]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])


def test_pairwise_scores_are_explicit_cosines():
    fake = FakeModel()
    adapter = JinaEmbeddingAdapter(
        JinaEmbeddingConfig(revision=REVISION), model_loader=lambda _: fake,
    )
    scores = adapter.pairwise_scores(["x"], ["x", "minus", "y"])
    np.testing.assert_allclose(scores, [[1.0, 0.0, 0.8]], atol=1e-6)


@pytest.mark.parametrize(
    "returned",
    [np.array([1.0, 2.0]), np.array([[0.0, 0.0]]), np.array([[np.nan, 1.0]])],
)
def test_adapter_rejects_invalid_model_outputs(returned):
    class InvalidModel:
        def encode(self, texts, **kwargs):
            return returned

    adapter = JinaEmbeddingAdapter(
        JinaEmbeddingConfig(revision=REVISION), model_loader=lambda _: InvalidModel(),
    )
    with pytest.raises(ValueError):
        adapter.encode(["x"])


def test_adapter_rejects_non_string_inputs_without_loading_model():
    adapter = JinaEmbeddingAdapter(
        JinaEmbeddingConfig(revision=REVISION),
        model_loader=lambda _: (_ for _ in ()).throw(AssertionError("must stay lazy")),
    )
    with pytest.raises(TypeError):
        adapter.encode([1])
