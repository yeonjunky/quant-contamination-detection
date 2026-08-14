"""Real-network, GPU-free tests for qcd.real_run — the shared core behind
scripts/run_pilot.py and scripts/run_main.py. Confirms the parts of the
real (non-mock) pipeline that don't need a GPU: item loading/capping,
candidate-code assembly, and that `run()` fails at exactly the expected
place (model loading) rather than somewhere earlier due to a wiring bug —
leaving items.parquet and manifest.json already written when it does.
"""

import datetime as dt

import pandas as pd
import pytest

from qcd.config import ModelSpec, Quant
from qcd.data.schema import Dataset, Item
from qcd.models.registry import QWEN2_5_7B
from qcd.real_run import RealRunConfig, _assemble_candidate_code, load_all_items, run


def _small_config(tmp_path, **overrides) -> RealRunConfig:
    defaults = dict(
        models=(QWEN2_5_7B,),
        quant_levels=(Quant.FP16,),
        output_dir=tmp_path,
        lcb_cutoff_boundary=dt.datetime(2023, 12, 1),
        lcb_release_version="release_v1",
        item_limit_per_condition=2,
    )
    defaults.update(overrides)
    return RealRunConfig(**defaults)


def test_load_all_items_respects_per_condition_cap(tmp_path):
    config = _small_config(tmp_path, item_limit_per_condition=2)
    items = load_all_items(config)

    counts: dict[Dataset, int] = {}
    for item in items:
        counts[item.dataset] = counts.get(item.dataset, 0) + 1

    for dataset, count in counts.items():
        assert count <= 2, f"{dataset} had {count} items, cap was 2"
    # All four conditions should be represented at this small scale.
    assert set(counts) == {Dataset.LCB_PRE, Dataset.LCB_POST, Dataset.HUMANEVAL, Dataset.MBPPPLUS}


def test_load_all_items_can_exclude_evalplus_datasets(tmp_path):
    config = _small_config(tmp_path, item_limit_per_condition=2, include_humaneval=False, include_mbppplus=False)
    items = load_all_items(config)
    datasets = {item.dataset for item in items}
    assert Dataset.HUMANEVAL not in datasets
    assert Dataset.MBPPPLUS not in datasets


def test_assemble_candidate_code_prepends_prompt_for_evalplus():
    item = Item(item_id="HumanEval/0", dataset=Dataset.HUMANEVAL, prompt="def f():\n    ")
    assert _assemble_candidate_code(item, "return 1\n") == "def f():\n    return 1\n"


def test_assemble_candidate_code_uses_completion_alone_for_lcb():
    item = Item(item_id="q1", dataset=Dataset.LCB_PRE, prompt="Solve this problem.")
    assert _assemble_candidate_code(item, "print('hi')\n") == "print('hi')\n"


def test_run_fails_at_model_loading_not_earlier(tmp_path, monkeypatch):
    # Do NOT actually call the real load_model() path here: on this profile
    # (no torch installed) it would try to download Qwen2.5-7B's tokenizer
    # over the network before failing for an unrelated reason (missing
    # torch), which is both slow/inappropriate for a test and not what this
    # test is checking. Stub load_model() to simulate "GPU path not ready
    # yet" directly, and confirm run()'s control flow reaches it (and only
    # it) after items/manifest are already written.
    import qcd.real_run as real_run_module

    def _stub_load_model(spec, quant, *, mock=False):
        raise NotImplementedError("real generation path — implement alongside scripts/run_smoke_test.py")

    monkeypatch.setattr(real_run_module, "load_model", _stub_load_model)

    config = _small_config(tmp_path, item_limit_per_condition=1)

    with pytest.raises(NotImplementedError, match="real generation path"):
        run(config)

    # Items and manifest should already be on disk — the failure happens
    # after those writes, not before.
    assert (tmp_path / "raw" / "items.parquet").exists()
    assert (tmp_path / "manifest.json").exists()
    df = pd.read_parquet(tmp_path / "raw" / "items.parquet")
    assert len(df) > 0
