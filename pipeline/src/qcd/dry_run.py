"""The local mock dry run: exercises the full paper §5 step-1→9 code path on
purely synthetic data, zero GPU/downloads, asserting the pipeline's named
invariants end-to-end rather than just "didn't crash."

Lives in the installed `qcd` package (not directly under `scripts/`) so
`tests/test_mock_pipeline_end_to_end.py` can import `run_dry_run` without
sys.path surgery; `scripts/run_dry_run.py` is a thin CLI shim over `main()`
here, satisfying pipeline_build_plan.md's "runnable both as a script and as
a pytest test" requirement without duplicating the pipeline logic.

Per pipeline_build_plan.md's smoke-test section, the mock harness bypasses
real sandboxed code execution entirely (MockModel's token streams decode to
GPT-2 vocabulary text, not runnable Python) — partial pass rate here comes
from `MockModel.partial_pass_rate()`, standing in for what
`scoring/pass_rate.py`'s real sandbox would produce.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import numpy as np

from qcd.analysis.aggregation import PooledSecondaryConditionsError, combined_items
from qcd.analysis.auc import empirical_auc
from qcd.analysis.logodds import spurious_pp_interaction
from qcd.config import Quant
from qcd.constants import (
    BASE_RATE_HUMANEVAL_ILLUSTRATIVE,
    BASE_RATE_LCB_POST_ILLUSTRATIVE,
    CDD_N_SAMPLES,
    CDD_SAMPLE_TEMPERATURE,
)
from qcd.data.schema import Dataset, Item
from qcd.detectors.cdd import peakedness
from qcd.detectors.mink_prob import mink_prob
from qcd.detectors.perplexity import negative_log_perplexity_score
from qcd.generation.cache import GenerationCache
from qcd.generation.sampler import sample_item
from qcd.io.raw_writer import RawDataWriter
from qcd.models.mock import MockModel
from qcd.pilot.pilot_report import (
    ValidationDiagnostics,
    base_rate,
    cohens_d_paired,
    pearson_r,
)

_MOCK_MODEL_NAME = "MockModel"
_PRECISIONS = (Quant.BF16.value, Quant.BNB_NF4.value)
_DETECTOR_NAMES = ("cdd", "perplexity", "mink_prob")


@dataclasses.dataclass
class InvariantChecks:
    logodds_matches_paper_table: bool
    pooling_guard_fires: bool
    contaminated_scores_higher_than_clean: dict[str, bool]  # per detector
    contaminated_has_higher_partial_pass: bool


@dataclasses.dataclass
class DryRunSummary:
    n_items: int
    diagnostics: ValidationDiagnostics
    invariants: InvariantChecks
    output_dir: Path


def build_synthetic_items(n_per_condition: int = 4, seed: int = 0) -> list[Item]:
    """~15-20 synthetic items (default 4 per condition x 4 conditions = 16)
    covering all four `Dataset` conditions."""
    items: list[Item] = []
    for dataset, prefix in (
        (Dataset.LCB_PRE, "lcb-pre"),
        (Dataset.LCB_POST, "lcb-post"),
        (Dataset.HUMANEVAL, "he"),
        (Dataset.MBPPPLUS, "mbpp"),
    ):
        for i in range(n_per_condition):
            item_id = f"{prefix}-{i}"
            items.append(Item(item_id=item_id, dataset=dataset, prompt=f"Solve synthetic problem {item_id}."))
    return items


def run_dry_run(
    output_dir: str | Path,
    *,
    n_per_condition: int = 4,
    n_cdd_samples: int = CDD_N_SAMPLES,
    seed: int = 0,
) -> DryRunSummary:
    output_dir = Path(output_dir)
    rng = random.Random(seed)
    items = build_synthetic_items(n_per_condition=n_per_condition, seed=seed)

    # --- register hidden ground truth on the mock, generate at both precisions ---
    model = MockModel()
    for item in items:
        contaminated = item.contamination_proxy
        quality = rng.uniform(0.85, 1.0) if contaminated else rng.uniform(0.2, 0.6)
        model.register_item(item.item_id, contaminated=contaminated, quality=quality)

    cache = GenerationCache(output_dir / "cache")
    writer = RawDataWriter(output_dir / "raw")
    writer.write_items(items)

    # detector_scores[(precision, detector)][item_id] = score
    detector_scores: dict[tuple[str, str], dict[str, float]] = {(p, d): {} for p in _PRECISIONS for d in _DETECTOR_NAMES}
    partial_pass_by_precision: dict[tuple[str, str], float] = {}

    for precision in _PRECISIONS:
        for item in items:
            generations = sample_item(
                model, cache, model_name=_MOCK_MODEL_NAME, quant=precision,
                item_id=item.item_id, prompt=item.prompt, n_samples=n_cdd_samples,
            )
            partial_pass = model.partial_pass_rate(item.item_id)
            partial_pass_by_precision[(precision, item.item_id)] = partial_pass
            prompt_logprobs = model.score_prompt_logprobs(item.item_id, item.prompt)

            writer.add_generation(
                model=_MOCK_MODEL_NAME, quant=precision, item_id=item.item_id, sample_id=0, is_greedy=True,
                text=generations.greedy.text, token_ids=generations.greedy.token_ids,
                token_logprobs=generations.greedy.token_logprobs, partial_pass_rate=partial_pass,
                prompt_token_logprobs=prompt_logprobs,
                decoding_temperature=0.0,
            )
            for sample_id, sample in enumerate(generations.samples, start=1):
                writer.add_generation(
                    model=_MOCK_MODEL_NAME, quant=precision, item_id=item.item_id, sample_id=sample_id,
                    is_greedy=False, text=sample.text, token_ids=sample.token_ids,
                    token_logprobs=sample.token_logprobs, decoding_temperature=CDD_SAMPLE_TEMPERATURE,
                )

            cdd_score = peakedness(generations.greedy.token_ids, [s.token_ids for s in generations.samples])
            ppl_score = negative_log_perplexity_score(prompt_logprobs)
            mink_score = mink_prob(prompt_logprobs)

            for detector, score in (("cdd", cdd_score), ("perplexity", ppl_score), ("mink_prob", mink_score)):
                writer.add_detector_score(model=_MOCK_MODEL_NAME, quant=precision, item_id=item.item_id, detector=detector, score=score)
                detector_scores[(precision, detector)][item.item_id] = score

            writer.add_detector_score(
                model=_MOCK_MODEL_NAME, quant=precision, item_id=item.item_id,
                detector="completion_perplexity",
                score=negative_log_perplexity_score(generations.greedy.token_logprobs),
            )
            writer.add_detector_score(
                model=_MOCK_MODEL_NAME, quant=precision, item_id=item.item_id,
                detector="completion_mink_prob",
                score=mink_prob(generations.greedy.token_logprobs),
            )

    writer.flush()

    # --- synthetic diagnostics: shape checks only, never study estimates ---
    report = ValidationDiagnostics()
    bf16, quant = _PRECISIONS
    item_ids = [item.item_id for item in items]
    labels = np.array([item.contamination_proxy for item in items])

    for detector in _DETECTOR_NAMES:
        before = [detector_scores[(bf16, detector)][item_id] for item_id in item_ids]
        after = [detector_scores[(quant, detector)][item_id] for item_id in item_ids]
        report.q1a_effect_size_d[detector] = cohens_d_paired(before, after)
        report.q1b_baseline_auc[detector] = empirical_auc(np.array(before), labels)
        report.q1b_cross_precision_r[detector] = pearson_r(before, after)

    for dataset in Dataset:
        outcomes = [
            partial_pass_by_precision[(bf16, item.item_id)]
            for item in items if item.dataset is dataset
        ]
        if outcomes:
            report.base_rates[(_MOCK_MODEL_NAME, dataset.value)] = base_rate(outcomes)

    # The synthetic mock's pass-rate process is intentionally precision
    # invariant, so its known Q2 interaction is zero and its paired outcomes
    # are perfectly correlated. These fields ensure the dry run exercises all
    # synthetic implementation diagnostics without fabricating a corpus-error estimate.
    report.q2_log_odds_effect = 0.0
    report.q2_item_level_r = pearson_r(
        [partial_pass_by_precision[(bf16, item_id)] for item_id in item_ids],
        [partial_pass_by_precision[(quant, item_id)] for item_id in item_ids],
    )

    # --- named invariants ---
    logodds_row = spurious_pp_interaction(
        0.50, humaneval_base_rate=BASE_RATE_HUMANEVAL_ILLUSTRATIVE, lcb_post_base_rate=BASE_RATE_LCB_POST_ILLUSTRATIVE,
    )
    logodds_ok = (
        abs(logodds_row["humaneval_drop_pp"] - 5.6) < 0.1
        and abs(logodds_row["lcb_post_drop_pp"] - 7.7) < 0.1
        and abs(logodds_row["spurious_interaction_pp"] - (-2.2)) < 0.1
    )

    try:
        combined_items([item for item in items if item.dataset in (Dataset.HUMANEVAL, Dataset.MBPPPLUS)])
        pooling_guard_fired = False
    except PooledSecondaryConditionsError:
        pooling_guard_fired = True

    contaminated_ids = [item.item_id for item in items if item.contamination_proxy]
    clean_ids = [item.item_id for item in items if not item.contamination_proxy]
    contaminated_higher = {}
    for detector in _DETECTOR_NAMES:
        bf16_scores = detector_scores[(bf16, detector)]
        mean_contaminated = float(np.mean([bf16_scores[i] for i in contaminated_ids]))
        mean_clean = float(np.mean([bf16_scores[i] for i in clean_ids]))
        contaminated_higher[detector] = mean_contaminated > mean_clean

    mean_pass_contaminated = float(np.mean([
        partial_pass_by_precision[(bf16, i)] for i in contaminated_ids
    ]))
    mean_pass_clean = float(np.mean([
        partial_pass_by_precision[(bf16, i)] for i in clean_ids
    ]))

    invariants = InvariantChecks(
        logodds_matches_paper_table=logodds_ok,
        pooling_guard_fires=pooling_guard_fired,
        contaminated_scores_higher_than_clean=contaminated_higher,
        contaminated_has_higher_partial_pass=(mean_pass_contaminated > mean_pass_clean),
    )

    return DryRunSummary(
        n_items=len(items),
        diagnostics=report,
        invariants=invariants,
        output_dir=output_dir,
    )


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="qcd_dry_run_") as tmp:
        summary = run_dry_run(tmp)

        print(f"Dry run: {summary.n_items} synthetic items, output written to {tmp}\n")
        print("Synthetic validation diagnostics (not study estimates):")
        print(f"  Q1a-shaped effect size d:     {summary.diagnostics.q1a_effect_size_d}")
        print(f"  Q1b-shaped baseline AUC:      {summary.diagnostics.q1b_baseline_auc}")
        print(f"  score-path correlation r:    {summary.diagnostics.q1b_cross_precision_r}")
        print(f"  Q2-shaped log-odds effect:    {summary.diagnostics.q2_log_odds_effect}")
        print(f"  item-level correlation r:    {summary.diagnostics.q2_item_level_r}")
        print(f"  synthetic base rates:        {summary.diagnostics.base_rates}")
        print()
        print("Invariant checks:")
        for field in dataclasses.fields(summary.invariants):
            print(f"  {field.name}: {getattr(summary.invariants, field.name)}")

        failures = []
        if not summary.invariants.logodds_matches_paper_table:
            failures.append("log-odds transform did not match paper's worked table")
        if not summary.invariants.pooling_guard_fires:
            failures.append("pooling guard did not fire on HumanEval+MBPP+ combination")
        if not all(summary.invariants.contaminated_scores_higher_than_clean.values()):
            failures.append(f"some detector did not score contaminated > clean: {summary.invariants.contaminated_scores_higher_than_clean}")
        if not summary.invariants.contaminated_has_higher_partial_pass:
            failures.append("contaminated items did not have a higher mean partial-pass rate than clean")

        if failures:
            print("\nFAILED invariant checks:")
            for f in failures:
                print(f"  - {f}")
            raise SystemExit(1)

        print("\nAll invariant checks passed.")


if __name__ == "__main__":
    main()
