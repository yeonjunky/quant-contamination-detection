"""E-F6: the smoke test has to be able to validate every model/precision the
main run will execute.

Paper §4.1's footprint table is the reference: ~14-16 GB bf16 / ~7-8 GB int8 /
~4-5 GB nf4 for the 7-8B arms, and ~64-65 / ~32 / ~18 GB for the 32B arms. A
single band per precision could not hold both size classes, so the band is
derived per model.
"""

import dataclasses
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


def test_awq_32b_band_never_exceeds_the_card():
    # Audit #5: the uncapped AWQ ceiling for a 32.5B model is 134 GB, so on an
    # 80 GB H100 any peak, even one leaving no room for the KV cache, passed.
    assert SMOKE.plausible_peak_gb(QWEN2_5_32B, Quant.GPTQ_AWQ_INT4)[1] == pytest.approx(134.0)
    lower, upper = SMOKE.plausible_peak_gb(QWEN2_5_32B, Quant.GPTQ_AWQ_INT4, device_total_gb=80.0)
    assert upper == 80.0
    assert not lower <= 85.0 <= upper


def test_device_cap_leaves_a_band_below_the_card_unchanged():
    assert SMOKE.plausible_peak_gb(QWEN2_5_7B, Quant.BNB_NF4, device_total_gb=80.0) == (
        SMOKE.plausible_peak_gb(QWEN2_5_7B, Quant.BNB_NF4)
    )


def _batch_size_from_cli(argv):
    args = SMOKE.build_parser().parse_args(argv)
    return SMOKE._sample_batch_size(args.sample_batch_size, SMOKE.get_model(args.model))


def test_sample_batch_size_defaults_to_the_registry_value():
    assert _batch_size_from_cli([]) == QWEN2_5_7B.sample_batch_size == 50


def test_an_unmeasured_32b_model_can_try_an_explicit_sample_batch_size():
    unmeasured = dataclasses.replace(QWEN2_5_32B, sample_batch_size=None)
    assert SMOKE._sample_batch_size(16, unmeasured) == 16
    with pytest.raises(SystemExit, match="pass --sample-batch-size"):
        SMOKE._sample_batch_size(None, unmeasured)
    assert _batch_size_from_cli(["--model", QWEN2_5_32B.name, "--sample-batch-size", "16"]) == 16
    with pytest.raises(SystemExit, match="must be >= 1"):
        _batch_size_from_cli(["--sample-batch-size", "0"])
