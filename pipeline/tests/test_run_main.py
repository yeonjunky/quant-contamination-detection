"""`scripts/run_main.py` is the only producer of main-study data, so it must
refuse to start unless the manifest's commit is the code that runs."""

import dataclasses
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_main.py"
_SPEC = importlib.util.spec_from_file_location("run_main", _SCRIPT)
RUN_MAIN = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(RUN_MAIN)


def test_a_dirty_checkout_stops_before_any_run(monkeypatch):
    started = []

    def _dirty():
        raise RuntimeError("main-study run refused: tracked files have uncommitted changes")

    monkeypatch.setattr(RUN_MAIN, "require_clean_checkout", _dirty)
    monkeypatch.setattr(RUN_MAIN, "run", lambda config: started.append(config))
    monkeypatch.setattr("sys.argv", ["run_main.py"])

    with pytest.raises(RuntimeError, match="uncommitted"):
        RUN_MAIN.main()
    assert started == []


def test_a_clean_checkout_starts_the_run(monkeypatch):
    started = []
    monkeypatch.setattr(RUN_MAIN, "require_clean_checkout", lambda: "0" * 40)
    monkeypatch.setattr(RUN_MAIN, "run", lambda config: started.append(config))
    monkeypatch.setattr("sys.argv", ["run_main.py"])
    monkeypatch.setattr(RUN_MAIN, "MAIN_ANALYSIS_MODELS", tuple(
        dataclasses.replace(model, sample_batch_size=model.sample_batch_size or 16)
        for model in RUN_MAIN.MAIN_ANALYSIS_MODELS
    ))

    RUN_MAIN.main()
    assert len(started) == 1


def test_an_unmeasured_sample_batch_size_stops_before_any_run(monkeypatch):
    started = []
    monkeypatch.setattr(RUN_MAIN, "require_clean_checkout", lambda: "0" * 40)
    monkeypatch.setattr(RUN_MAIN, "run", lambda config: started.append(config))
    monkeypatch.setattr("sys.argv", ["run_main.py"])
    measured, *rest = RUN_MAIN.MAIN_ANALYSIS_MODELS
    roster = (dataclasses.replace(measured, sample_batch_size=None), *(
        dataclasses.replace(model, sample_batch_size=model.sample_batch_size or 16) for model in rest
    ))
    monkeypatch.setattr(RUN_MAIN, "MAIN_ANALYSIS_MODELS", roster)

    with pytest.raises(ValueError, match="Measure it on the H100") as refused:
        RUN_MAIN.main()
    assert started == []
    assert str(refused.value).startswith(f"['{measured.name}'] have no sample_batch_size")


@pytest.fixture
def started(monkeypatch):
    configs = []
    monkeypatch.setattr(RUN_MAIN, "require_clean_checkout", lambda: "0" * 40)
    monkeypatch.setattr(RUN_MAIN, "run", lambda config: configs.append(config))
    monkeypatch.setattr(RUN_MAIN, "MAIN_ANALYSIS_MODELS", tuple(
        dataclasses.replace(model, sample_batch_size=model.sample_batch_size or 16)
        for model in RUN_MAIN.MAIN_ANALYSIS_MODELS
    ))
    return configs


def test_the_driver_runs_the_frozen_settings(monkeypatch, started):
    from qcd.constants import CDD_N_SAMPLES, LCB_SHARED_CONTROL_BOUNDARY

    monkeypatch.setattr("sys.argv", ["run_main.py"])
    RUN_MAIN.main()

    (config,) = started
    assert config.cells is None
    assert config.n_cdd_samples == CDD_N_SAMPLES == 50
    assert config.lcb_cutoff_boundary.date().isoformat() == LCB_SHARED_CONTROL_BOUNDARY
    assert config.lcb_release_version == "release_v6"
    assert config.output_dir == Path(__file__).resolve().parents[2] / "data" / "raw" / "main"


@pytest.mark.parametrize("flag", [
    ["--n-cdd-samples", "5"], ["--lcb-cutoff", "2024-01-01"],
    ["--lcb-release", "release_v5"], ["--output-dir", "/tmp/elsewhere"],
])
def test_frozen_settings_have_no_override(monkeypatch, started, flag):
    monkeypatch.setattr("sys.argv", ["run_main.py", *flag])
    with pytest.raises(SystemExit):
        RUN_MAIN.main()
    assert started == []


def test_cells_select_a_subset_of_the_full_study(monkeypatch, started):
    from qcd.config import Quant

    monkeypatch.setattr("sys.argv", [
        "run_main.py",
        "--cell", "Qwen2.5-7B-Instruct:bnb_nf4",
        "--cell", "Olmo3.1-32B-Instruct:bf16",
    ])
    RUN_MAIN.main()

    (config,) = started
    assert config.cells == {
        ("Qwen2.5-7B-Instruct", Quant.BNB_NF4), ("Olmo3.1-32B-Instruct", Quant.BF16),
    }
    # The study the manifest describes is still all five models x four levels.
    assert [m.name for m in config.models] == [m.name for m in RUN_MAIN.MAIN_ANALYSIS_MODELS]
    assert len(config.models) == 5
    assert config.quant_levels == RUN_MAIN.MAIN_QUANT_LEVELS
    assert len(config.quant_levels) == 4


@pytest.mark.parametrize(
    "cell", ["Qwen2.5-7B:bf16", "Qwen2.5-7B-Instruct:fp8", "Qwen2.5-7B-Instruct"]
)
def test_an_unknown_cell_stops_before_any_run(monkeypatch, started, cell):
    checked = []
    monkeypatch.setattr(RUN_MAIN, "require_clean_checkout", lambda: checked.append(1))
    monkeypatch.setattr("sys.argv", ["run_main.py", "--cell", cell])
    with pytest.raises(SystemExit):
        RUN_MAIN.main()
    assert started == []
    assert checked == []
