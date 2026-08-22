"""Regression coverage for the explicitly forced BF16 baseline."""

import torch
import transformers

from qcd.config import Quant
from qcd.models.loader import _load_bf16
from qcd.models.registry import QWEN2_5_7B


def test_bf16_loader_forces_torch_bfloat16(monkeypatch):
    captured = {}
    fake_tokenizer = object()
    fake_model = object()

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda repo_id, **kwargs: fake_tokenizer,
    )

    def fake_model_from_pretrained(repo_id, **kwargs):
        captured.update(kwargs)
        return fake_model

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_from_pretrained,
    )

    adapter = _load_bf16(QWEN2_5_7B, Quant.BF16)
    assert adapter.model is fake_model
    assert adapter.tokenizer is fake_tokenizer
    assert captured == {
        "revision": QWEN2_5_7B.revision,
        "dtype": torch.bfloat16,
        "device_map": "auto",
    }
    assert adapter.revision == QWEN2_5_7B.revision
