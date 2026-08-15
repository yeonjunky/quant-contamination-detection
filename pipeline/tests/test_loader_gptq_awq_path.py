"""Mock-profile-safe coverage for models/loader.py's GPTQ/AWQ (llm-compressor)
path: confirms the missing-checkpoint error is clear and actionable, and that
it's raised before any heavy (torch/transformers) import — the directory
existence check in `_load_gptq_or_awq` happens first, mirroring the lazy-
import discipline the module docstring describes.
"""

import pytest

import qcd.models.loader as loader
from qcd.config import Quant
from qcd.models.loader import _quantized_checkpoint_dir, load_model
from qcd.models.registry import QWEN2_5_7B


@pytest.fixture(autouse=True)
def _isolated_quantized_dir(tmp_path, monkeypatch):
    # loader._QUANTIZED_DIR resolves to the real repo-root data/quantized/
    # by default — on a machine that's actually run scripts/quantize_model.py
    # (e.g. this one, 2026-08-15), the canonical checkpoint genuinely exists,
    # which would make these "no checkpoint yet" tests fail depending on
    # ambient filesystem state rather than the code under test. Point at an
    # empty tmp_path instead, so the tests are deterministic everywhere.
    monkeypatch.setattr(loader, "_QUANTIZED_DIR", tmp_path)


def test_gptq_awq_load_raises_file_not_found_when_no_checkpoint():
    with pytest.raises(FileNotFoundError, match="quantize_model.py"):
        load_model(QWEN2_5_7B, Quant.GPTQ_AWQ_INT4, mock=False)


def test_gptq_awq_error_names_the_expected_checkpoint_path():
    expected_dir = _quantized_checkpoint_dir(QWEN2_5_7B)
    with pytest.raises(FileNotFoundError, match=str(expected_dir)):
        load_model(QWEN2_5_7B, Quant.GPTQ_AWQ_INT4, mock=False)


def test_quantized_checkpoint_dir_has_no_calibration_suffix():
    # The canonical loader.py path is calibration-agnostic — quantize_model.py
    # saves calibration-tagged variants (-awq-code/-awq-chat) elsewhere; only
    # a deliberate copy/re-quantize step ever populates this exact path.
    path = _quantized_checkpoint_dir(QWEN2_5_7B)
    assert path.name == f"{QWEN2_5_7B.name}-awq"
    assert "code" not in path.name
    assert "chat" not in path.name
