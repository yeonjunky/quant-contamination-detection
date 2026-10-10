"""`scripts/check_sandbox_pool.py`: an item fails when either scoring path
does not give its reference solution a full score."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_SCRIPT = Path(__file__).parents[1] / "scripts" / "check_sandbox_pool.py"
_SPEC = importlib.util.spec_from_file_location("check_sandbox_pool", _SCRIPT)
CHECK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CHECK)

_ITEMS = [SimpleNamespace(item_id=f"i{n}") for n in range(3)]


def test_all_full_scores_pass():
    assert CHECK.compare(_ITEMS, lambda items: [1.0] * len(items), lambda item: 1.0) == []


def test_a_pooled_score_below_one_fails_that_item():
    pooled = lambda items: [1.0, 0.5, 1.0]  # noqa: E731
    assert CHECK.compare(_ITEMS, pooled, lambda item: 1.0) == ["i1: pooled 0.5, serial 1.0"]


def test_a_serial_score_below_one_fails_that_item():
    serial = lambda item: 0.0 if item.item_id == "i2" else 1.0  # noqa: E731
    assert CHECK.compare(_ITEMS, lambda items: [1.0] * 3, serial) == ["i2: pooled 1.0, serial 0.0"]
