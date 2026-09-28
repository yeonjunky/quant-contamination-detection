"""`scripts/run_main.py` is the only producer of main-study data, so it must
refuse to start unless the manifest's commit is the code that runs."""

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

    RUN_MAIN.main()
    assert len(started) == 1
