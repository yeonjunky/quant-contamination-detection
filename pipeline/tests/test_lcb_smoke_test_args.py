"""`scripts/run_lcb_smoke_test.py` must be able to validate every arm the main
run will execute, not just Qwen2.5-7B at nf4.

Same requirement as `tests/test_smoke_test_scope.py` states for
`scripts/run_smoke_test.py`: a precision that cannot be selected here would
first be loaded on this code path during the main run itself.
"""

import importlib.util
from pathlib import Path

from qcd.config import Quant
from qcd.models.registry import ALL_MODELS, OLMO3_1_32B, QWEN2_5_7B, QWEN2_5_32B

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_lcb_smoke_test.py"
_SPEC = importlib.util.spec_from_file_location("run_lcb_smoke_test", _SCRIPT)
LCB = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(LCB)


def test_every_precision_in_the_ladder_is_selectable():
    assert set(LCB._QUANT_CHOICES) == {quant.value for quant in Quant}


def test_defaults_are_unchanged():
    # The previous hard-coded arm, now a default rather than a constant.
    args = LCB.build_parser().parse_args([])
    assert args.model == QWEN2_5_7B.name
    assert args.quant == Quant.BNB_NF4.value
    assert args.lcb_release == "release_v6"


def test_model_and_quant_are_selectable():
    args = LCB.build_parser().parse_args(
        ["--model", QWEN2_5_32B.name, "--quant", Quant.BNB_INT8.value]
    )
    assert args.model == QWEN2_5_32B.name
    assert args.quant == Quant.BNB_INT8.value


def test_every_registry_model_resolves_through_the_argument():
    for spec in ALL_MODELS:
        assert LCB.get_model(spec.name) is spec
    assert LCB.get_model(OLMO3_1_32B.name) is OLMO3_1_32B


def test_each_arm_writes_to_its_own_default_directory():
    """Every arm is validated with the default output directory, so a shared
    default left only the last arm's report on disk."""
    arms = [
        LCB._parse_args(["--model", spec.name, "--quant", quant.value])
        for spec in ALL_MODELS for quant in Quant
    ]
    directories = {args.output_dir for args in arms}
    assert len(directories) == len(arms)
    for directory in directories:
        assert directory.parent == LCB._DEFAULT_OUTPUT_ROOT

    explicit = LCB._parse_args(["--output-dir", "/tmp/elsewhere"])
    assert str(explicit.output_dir) == "/tmp/elsewhere"
