"""E-F6: the smoke test has to be able to validate every model/precision the
main run will execute.

Paper §4.1's footprint table is the reference: ~14-16 GB bf16 / ~7-8 GB int8 /
~4-5 GB nf4 for the 7-8B arms, and ~64-65 / ~32 / ~18 GB for the 32B arms. A
single band per precision could not hold both size classes, so the band is
derived per model.
"""

import importlib.util
from pathlib import Path

import pytest

from qcd.config import Quant
from qcd.models.registry import OLMO3_1_32B, QWEN2_5_7B, QWEN2_5_32B

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_smoke_test.py"
_SPEC = importlib.util.spec_from_file_location("run_smoke_test", _SCRIPT)
SMOKE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(SMOKE)


def test_every_precision_in_the_ladder_is_selectable():
    # Paper §4.3's four rungs; bf16 and int8 used to be unreachable here, which
    # meant their first real load would have happened in the main run.
    assert set(SMOKE._QUANT_CHOICES) == {quant.value for quant in Quant}


@pytest.mark.parametrize(
    ("spec", "quant", "observed_gb"),
    [
        (QWEN2_5_7B, Quant.BF16, 15.0),
        (QWEN2_5_7B, Quant.BNB_INT8, 7.5),
        (QWEN2_5_7B, Quant.BNB_NF4, 4.5),
        # Measured 2026-08-15: plain-transformers AWQ inference peaks near the
        # bf16 footprint, not the int4 one.
        (QWEN2_5_7B, Quant.GPTQ_AWQ_INT4, 15.5),
        (QWEN2_5_32B, Quant.BF16, 64.5),
        (QWEN2_5_32B, Quant.BNB_INT8, 32.0),
        (QWEN2_5_32B, Quant.BNB_NF4, 18.0),
        (OLMO3_1_32B, Quant.BNB_NF4, 18.0),
    ],
)
def test_band_accepts_the_paper_footprint(spec, quant, observed_gb):
    lower, upper = SMOKE.plausible_peak_gb(spec, quant)
    assert lower <= observed_gb <= upper


def test_band_still_catches_a_precision_mistake():
    # The failure this band exists for: asking for nf4 and silently getting
    # bf16 weights (~8x the footprint).
    lower, upper = SMOKE.plausible_peak_gb(QWEN2_5_7B, Quant.BNB_NF4)
    bf16_peak = QWEN2_5_7B.param_count_b * 2.0
    assert bf16_peak > upper


def test_band_is_model_specific_not_precision_only():
    small = SMOKE.plausible_peak_gb(QWEN2_5_7B, Quant.BNB_NF4)
    large = SMOKE.plausible_peak_gb(QWEN2_5_32B, Quant.BNB_NF4)
    assert large[1] > small[1]
