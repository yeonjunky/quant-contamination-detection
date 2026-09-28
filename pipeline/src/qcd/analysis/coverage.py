"""Synthetic-data interval-coverage check for β_QE — paper §4.5.5's
"Verify interval coverage on synthetic data before the main run."

§4.5.5 fixes the shape of the check: generate from the design's own shape
(one model's item counts, e.g. 690 `possible-exposure` and 182
`shared-clean-control`; a normal item-difficulty distribution; a known
β_QE), then estimate each candidate method's achieved coverage over enough
replications to separate it from the nominal 95%. The generating parameters,
the replication count, and every method's achieved coverage go into the
analysis manifest. **That check fixes which method supplies the reported
interval**, not the main-study output.

This module is deliberately generic over the method: `INTERVAL_METHODS` maps
a name to a callable, so an MCMC or full-joint-Hessian Laplace interval
(§4.5.5's two permitted alternatives) can be added and held to the same
check. `variational_bayes_posterior_sd` is included precisely because
§4.5.5 rules it out as an interval — keeping it in the comparison is what
makes that ruling checkable rather than asserted.

Nothing here touches real data, a GPU, or the network.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from qcd.analysis.conditional_logit import ConditionalLogitResult, fit_conditional_logit_beta_qe
from qcd.analysis.logodds import solve_intercept_for_base_rate
from qcd.constants import BASE_RATE_LCB_POST_ILLUSTRATIVE, DIFFICULTY_SIGMA

#: Qwen2.5-32B-Instruct's LCB label-group sizes (§4.5.6).
DESIGN_N_POSSIBLE_EXPOSURE = 690
DESIGN_N_SHARED_CLEAN_CONTROL = 182

# --------------------------------------------------------------------------
# Where the check's own record lives
# --------------------------------------------------------------------------
#
# §4.5.5 makes the coverage check a pre-main-run artifact: it "fixes which
# method supplies the reported interval", and the analysis manifest has to
# carry its generating parameters, replication count and achieved coverage.
# A record that only ever existed in a temp directory cannot do that, so the
# canonical copy is a tracked file in the repository and both
# `scripts/verify_interval_coverage.py` (writer) and
# `scripts/run_analysis.py` (reader) address it through these constants.

#: `pipeline/analysis_artifacts/` — analysis inputs/records that are part of
#: the protocol, as opposed to run outputs, which live with their run under
#: the gitignored `data/` tree.
ANALYSIS_ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "analysis_artifacts"
INTERVAL_COVERAGE_RECORD_FILENAME = "interval_coverage_check.json"
INTERVAL_COVERAGE_RECORD_PATH = ANALYSIS_ARTIFACTS_DIR / INTERVAL_COVERAGE_RECORD_FILENAME
#: What `record` field a valid coverage artifact carries.
INTERVAL_COVERAGE_RECORD_ID = "paper_4_5_5_interval_coverage_check"
#: Re-run this to regenerate the artifact (documented in the analysis manifest
#: so the record is reproducible from the manifest alone).
INTERVAL_COVERAGE_COMMAND = (
    "python scripts/verify_interval_coverage.py --replications 200 "
    "--beta-qe 0.0 0.5 --seed 20250918"
)


def load_interval_coverage_record(path: str | Path | None = None) -> dict:
    """Read the §4.5.5 coverage record, refusing one that does not carry what
    the analysis manifest must report.

    §4.5.5: "Record the generating parameters, the number of replications,
    and the achieved coverage of every method examined in the analysis
    manifest." Each of those three is checked here, so a truncated or
    hand-edited record fails at read time rather than producing a manifest
    that silently omits them.
    """
    path = Path(path) if path is not None else INTERVAL_COVERAGE_RECORD_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"no interval-coverage record at {path}. Paper §4.5.5 requires the coverage "
            "check before the main run, and its result in the analysis manifest. "
            f"Regenerate it with: {INTERVAL_COVERAGE_COMMAND}"
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("record") != INTERVAL_COVERAGE_RECORD_ID:
        raise ValueError(
            f"{path} is not a {INTERVAL_COVERAGE_RECORD_ID} record "
            f"(found record={record.get('record')!r})"
        )
    scenarios = record.get("scenarios")
    if not scenarios:
        raise ValueError(f"{path} records no coverage scenario")
    for index, scenario in enumerate(scenarios):
        for field in ("design", "n_replications", "coverage"):
            if not scenario.get(field):
                raise ValueError(f"{path} scenario {index} is missing {field!r} (§4.5.5)")
    if not record.get("methods_examined"):
        raise ValueError(f"{path} does not list the methods examined (§4.5.5)")
    return record


def interval_coverage_record_digest(path: str | Path | None = None) -> str:
    """sha256 of the record file as stored, so a manifest can name the exact
    bytes it summarized."""
    path = Path(path) if path is not None else INTERVAL_COVERAGE_RECORD_PATH
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclasses.dataclass(frozen=True)
class SyntheticDesign:
    """Generating parameters for one synthetic replication, recorded
    verbatim in the coverage output so §4.5.5's manifest requirement can be
    satisfied from the artifact alone."""

    n_possible_exposure: int = DESIGN_N_POSSIBLE_EXPOSURE
    n_shared_clean_control: int = DESIGN_N_SHARED_CLEAN_CONTROL
    difficulty_sd: float = DIFFICULTY_SIGMA
    marginal_base_rate: float = BASE_RATE_LCB_POST_ILLUSTRATIVE
    beta_q: float = -0.4
    beta_e: float = 0.3
    beta_qe: float = 0.5
    baseline: str = "bf16"
    target: str = "bnb_nf4"

    def as_dict(self) -> dict:
        payload = dataclasses.asdict(self)
        payload["intercept"] = self.intercept
        return payload

    @property
    def intercept(self) -> float:
        """Condition intercept whose marginal (β=0) accuracy equals
        `marginal_base_rate` at `difficulty_sd` — reuses
        `analysis.logodds.solve_intercept_for_base_rate` so the base rate
        means the same thing here as in §4.5.3's worked table."""
        return solve_intercept_for_base_rate(self.marginal_base_rate, self.difficulty_sd)


def simulate_design_dataset(design: SyntheticDesign, rng: np.random.Generator) -> pd.DataFrame:
    """One synthetic replication shaped like a single model's Q2 cell.

    Item difficulty is drawn ``N(0, difficulty_sd)``; each item is measured
    once at `baseline` (Q=0) and once at `target` (Q=1); the outcome is
    Bernoulli with
    ``logit p = intercept + difficulty_i + β_Q Q + β_E E + β_QE Q E``
    under §3.1's coding. Returns a tidy frame with columns
    ``item_id, precision, exposure_proxy, correct``.
    """
    n_items = design.n_possible_exposure + design.n_shared_clean_control
    exposure = np.concatenate(
        [np.ones(design.n_possible_exposure), np.zeros(design.n_shared_clean_control)]
    )
    difficulty = rng.normal(0.0, design.difficulty_sd, n_items)
    intercept = design.intercept

    frames = []
    for precision, q in ((design.baseline, 0.0), (design.target, 1.0)):
        logit = (
            intercept
            + difficulty
            + design.beta_q * q
            + design.beta_e * exposure
            + design.beta_qe * q * exposure
        )
        p = 1.0 / (1.0 + np.exp(-logit))
        frames.append(
            pd.DataFrame(
                {
                    "item_id": np.arange(n_items),
                    "precision": precision,
                    "exposure_proxy": exposure.astype(bool),
                    "correct": (rng.random(n_items) < p).astype(int),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Candidate interval methods
# --------------------------------------------------------------------------

#: An interval method maps one replication's frame plus the design to
#: ``(point_estimate, ci_low, ci_high)`` or ``None`` when not estimable.
IntervalMethod = Callable[[pd.DataFrame, SyntheticDesign], tuple[float, float, float] | None]


def conditional_logit_wald(
    df: pd.DataFrame, design: SyntheticDesign
) -> tuple[float, float, float] | None:
    """§4.5.5's adopted method: item-stratified conditional logistic Wald."""
    result: ConditionalLogitResult = fit_conditional_logit_beta_qe(
        df, baseline=design.baseline, target=design.target
    )
    if result.status != "computed" or result.ci_low is None or result.ci_high is None:
        return None
    return (float(result.beta_qe), float(result.ci_low), float(result.ci_high))


def variational_bayes_posterior_sd(
    df: pd.DataFrame, design: SyntheticDesign
) -> tuple[float, float, float] | None:
    """The method §4.5.5 **rules out**: mean-field VB posterior mean ±1.96
    posterior SD from `BinomialBayesMixedGLM.fit_vb`. Kept in the comparison
    so its undercoverage is measured rather than asserted."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM  # noqa: PLC0415

    working = df.copy()
    working["item_id"] = working["item_id"].astype(str)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glmm = BinomialBayesMixedGLM.from_formula(
                "correct ~ precision * exposure_proxy",
                {"item": "0 + C(item_id)"},
                working,
            )
            fitted = glmm.fit_vb()
    except Exception:  # noqa: BLE001
        return None
    names = list(glmm.fep_names)
    interaction = [i for i, name in enumerate(names) if ":" in name]
    if len(interaction) != 1:
        return None
    index = interaction[0]
    mean = float(fitted.fe_mean[index])
    sd = float(fitted.fe_sd[index])
    if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
        return None
    # patsy's reference level is the alphabetically first precision; with
    # bf16 < bnb_nf4 that is bf16, matching §3.1's Q=0 for bf16.
    return (mean, mean - 1.959963984540054 * sd, mean + 1.959963984540054 * sd)


INTERVAL_METHODS: dict[str, IntervalMethod] = {
    "conditional_logit_wald": conditional_logit_wald,
    "variational_bayes_posterior_sd": variational_bayes_posterior_sd,
}


# --------------------------------------------------------------------------
# Coverage estimation
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CoverageResult:
    method: str
    true_beta_qe: float
    n_replications: int
    n_estimable: int
    n_covered: int
    coverage: float | None
    coverage_monte_carlo_se: float | None
    mean_interval_width: float | None
    mean_point_estimate: float | None
    empirical_sd_of_point_estimate: float | None
    mean_reported_half_width: float | None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _summarize(
    method: str,
    true_beta_qe: float,
    n_replications: int,
    points: list[float],
    lows: list[float],
    highs: list[float],
) -> CoverageResult:
    n_estimable = len(points)
    if n_estimable == 0:
        return CoverageResult(
            method=method,
            true_beta_qe=true_beta_qe,
            n_replications=n_replications,
            n_estimable=0,
            n_covered=0,
            coverage=None,
            coverage_monte_carlo_se=None,
            mean_interval_width=None,
            mean_point_estimate=None,
            empirical_sd_of_point_estimate=None,
            mean_reported_half_width=None,
        )
    point = np.asarray(points, dtype=float)
    low = np.asarray(lows, dtype=float)
    high = np.asarray(highs, dtype=float)
    covered = (low <= true_beta_qe) & (true_beta_qe <= high)
    coverage = float(covered.mean())
    width = high - low
    return CoverageResult(
        method=method,
        true_beta_qe=true_beta_qe,
        n_replications=n_replications,
        n_estimable=n_estimable,
        n_covered=int(covered.sum()),
        coverage=coverage,
        coverage_monte_carlo_se=float(np.sqrt(coverage * (1 - coverage) / n_estimable)),
        mean_interval_width=float(width.mean()),
        mean_point_estimate=float(point.mean()),
        empirical_sd_of_point_estimate=(
            float(point.std(ddof=1)) if n_estimable > 1 else None
        ),
        mean_reported_half_width=float((width / 2).mean()),
    )


def estimate_interval_coverage(
    *,
    design: SyntheticDesign,
    n_replications: int,
    methods: dict[str, IntervalMethod] | None = None,
    seed: int = 0,
) -> dict[str, CoverageResult]:
    """Achieved coverage of each candidate interval method for β_QE.

    Every method sees the **same** replications (one shared seed sequence),
    so their coverages are directly comparable. A replication a method
    cannot estimate is excluded from that method's denominator and counted
    in ``n_replications - n_estimable``.
    """
    if n_replications < 1:
        raise ValueError("n_replications must be at least 1")
    methods = dict(methods or INTERVAL_METHODS)
    rng = np.random.default_rng(seed)
    collected: dict[str, tuple[list[float], list[float], list[float]]] = {
        name: ([], [], []) for name in methods
    }
    for _ in range(n_replications):
        df = simulate_design_dataset(design, rng)
        for name, method in methods.items():
            outcome = method(df, design)
            if outcome is None:
                continue
            point, low, high = outcome
            collected[name][0].append(point)
            collected[name][1].append(low)
            collected[name][2].append(high)
    return {
        name: _summarize(name, design.beta_qe, n_replications, *collected[name])
        for name in methods
    }
