"""The four confirmatory tests C1-C4 and their Holm correction — paper
§4.5.1 (Q1a estimand), §4.5.2 (Q1b AUC comparison) and §4.5.6 (the family,
the C4 intersection-union construction, the DeLong covariance, and the
not-estimable rule).

Scope, restated from §4.5.6 so it cannot drift: the family is fixed at four
tests, all on **Qwen2.5-32B-Instruct** and the **bf16 -> BNB-nf4** contrast.

- C1, C2, C3 (Q1a): one two-sided paired t-test per detector (perplexity,
  Min-k% Prob, CDD) over **all** LCB release_v6 items for that model —
  1,055 for Qwen, including the 183 intermediate-date items. The estimand is
  the mean within-item ``nf4 - bf16`` score difference. No exposure label is
  used, so `possible-exposure` / `shared-clean-control` / intermediate items
  all enter the same paired sample.
- C4 (Q1b): at each precision p, ``g_p = (AUC_perplexity,p + AUC_Min-k,p)/2
  - AUC_CDD,p`` on the 690 `possible-exposure` vs. 182
  `shared-clean-control` items. This is an equal-weight mean of two AUCs, not
  an AUC of pooled raw scores; the component AUCs are reported too. A
  reversal requires ``g_bf16 > 0 and g_nf4 < 0`` or ``g_bf16 < 0 and
  g_nf4 > 0`` — a gap change alone is not a reversal.

**Score orientation.** Every score passed in must already be oriented so that
larger = more possible exposure (negative log perplexity, unmodified Min-k
log-probability, CDD peakedness). `qcd.detectors` already stores all three
that way; this module never flips a sign.

**Not estimable.** An empty label group, missing required scores, or a
zero/undefined variance is reported with ``p_value = 1.0`` and a
``status`` naming the reason. The slot is retained for multiplicity
accounting rather than dropped or replaced (§4.5.6).

**Dependency note.** `analysis/_stats.py` deliberately avoids scipy, but that
rule is scoped to hand-reproducing the paper's own worked power/AUC tables,
where the failure mode is re-deriving a citation number by an inconsistent
path. A confirmatory t-test and normal Wald tail on real experimental data
have no hand-verifiable target to match; hand-rolling a Student-t CDF would
be exactly the kind of unverified numerical code that discipline warns
against. scipy's `stats.t` / `stats.norm` are used for the tail
probabilities only, and `tests/test_confirmatory.py` checks the paired test
against `scipy.stats.ttest_rel` end to end.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from qcd.constants import ALPHA

#: The four confirmatory slots, in the paper's own order (§4.5.6).
CONFIRMATORY_SLOTS = ("C1", "C2", "C3", "C4")

#: §4.5.6's detector-to-slot assignment for Q1a. Order matters: Holm is
#: applied to the four p-values in this order, and the mapping is fixed
#: before the main run.
Q1A_SLOT_DETECTORS = {"C1": "perplexity", "C2": "mink_prob", "C3": "cdd"}

#: Detector names used by the C4 gap g_p = mean(probability family) - CDD.
C4_PROBABILITY_DETECTORS = ("perplexity", "mink_prob")
C4_REFERENCE_DETECTOR = "cdd"

STATUS_COMPUTED = "computed"
STATUS_MISSING_SCORES = "not_estimable_missing_scores"
STATUS_INSUFFICIENT_PAIRS = "not_estimable_insufficient_pairs"
STATUS_ZERO_VARIANCE = "not_estimable_zero_variance"
STATUS_EMPTY_LABEL_GROUP = "not_estimable_empty_label_group"


# --------------------------------------------------------------------------
# C1-C3: paired detector-score shift (§4.5.1)
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PairedShiftResult:
    """One of C1-C3. `mean_difference` is target minus baseline, i.e. the
    nf4 - bf16 within-item shift; `cohens_dz` is that mean divided by the SD
    of the item differences (§4.5.1's d_z, not a pooled-SD d)."""

    detector: str
    status: str
    n_pairs: int
    mean_difference: float | None
    sd_difference: float | None
    standard_error: float | None
    t_statistic: float | None
    degrees_of_freedom: int | None
    p_value: float
    cohens_dz: float | None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def paired_score_shift_test(
    baseline_scores: Sequence[float] | np.ndarray,
    target_scores: Sequence[float] | np.ndarray,
    *,
    detector: str,
) -> PairedShiftResult:
    """Two-sided paired t-test of the within-item ``target - baseline``
    detector-score difference (§4.5.1, §4.5.6 C1-C3).

    Both arrays must be aligned item-by-item and the same length. A
    non-finite entry anywhere, fewer than two pairs, or a zero difference
    variance yields ``p_value = 1.0`` with the corresponding not-estimable
    status rather than an exception.
    """
    from scipy import stats  # noqa: PLC0415  (see module docstring)

    before = np.asarray(baseline_scores, dtype=float)
    after = np.asarray(target_scores, dtype=float)
    if before.shape != after.shape:
        raise ValueError(
            f"paired_score_shift_test needs aligned arrays, got {before.shape} and {after.shape}"
        )
    n = int(before.size)

    def not_estimable(status: str) -> PairedShiftResult:
        return PairedShiftResult(
            detector=detector,
            status=status,
            n_pairs=n,
            mean_difference=None,
            sd_difference=None,
            standard_error=None,
            t_statistic=None,
            degrees_of_freedom=None,
            p_value=1.0,
            cohens_dz=None,
        )

    if n and not (np.isfinite(before).all() and np.isfinite(after).all()):
        return not_estimable(STATUS_MISSING_SCORES)
    if n < 2:
        return not_estimable(STATUS_INSUFFICIENT_PAIRS)

    diff = after - before
    sd = float(diff.std(ddof=1))
    mean = float(diff.mean())
    if sd == 0.0:
        return dataclasses.replace(
            not_estimable(STATUS_ZERO_VARIANCE),
            mean_difference=mean,
            sd_difference=0.0,
        )

    se = sd / np.sqrt(n)
    t_stat = mean / se
    df = n - 1
    p_value = float(2 * stats.t.sf(abs(t_stat), df))
    return PairedShiftResult(
        detector=detector,
        status=STATUS_COMPUTED,
        n_pairs=n,
        mean_difference=mean,
        sd_difference=sd,
        standard_error=float(se),
        t_statistic=float(t_stat),
        degrees_of_freedom=int(df),
        p_value=p_value,
        cohens_dz=float(mean / sd),
    )


# --------------------------------------------------------------------------
# DeLong covariance of several AUCs on the same items (§4.5.6)
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DeLongResult:
    """Joint DeLong estimate of several AUCs measured on one shared item set
    with one shared label vector. `covariance` is the K x K asymptotic
    covariance of the AUC estimates; any linear contrast's SE is
    ``sqrt(c @ covariance @ c)``. §4.5.6 notes that a singular covariance is
    not itself a problem — no inverse is taken here."""

    names: tuple[str, ...]
    aucs: np.ndarray
    covariance: np.ndarray
    n_positive: int
    n_negative: int

    def auc(self, name: str) -> float:
        return float(self.aucs[self.names.index(name)])

    def contrast(self, weights: dict[str, float]) -> tuple[float, float]:
        """Point estimate and standard error of ``sum_k w_k * AUC_k``."""
        c = np.zeros(len(self.names))
        for name, weight in weights.items():
            c[self.names.index(name)] = weight
        variance = float(c @ self.covariance @ c)
        estimate = float(c @ self.aucs)
        se = float(np.sqrt(variance)) if variance > 0 else 0.0
        return estimate, se


def _psi_matrix(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """DeLong's kernel psi(X, Y) = 1 if X > Y, 0.5 if X == Y, 0 otherwise,
    as a full ``len(positive) x len(negative)`` matrix. Computed directly
    from the definition rather than via the midrank shortcut: the item
    counts here (at most ~1,000 per group) make the O(mn) form cheap, and it
    handles ties without a separate rank-adjustment step."""
    diff = positive[:, None] - negative[None, :]
    return (diff > 0).astype(float) + 0.5 * (diff == 0)


def delong_auc_covariance(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    names: Sequence[str],
) -> DeLongResult:
    """DeLong (1988) covariance of K AUCs computed on the same items.

    `scores` is ``(K, n_items)`` — one row per AUC estimate (detector x
    precision) — and `labels` is a length-``n_items`` boolean vector, True
    for `possible-exposure`. Scores must already be oriented so larger =
    more possible exposure.

    The estimator is the standard structural-component form: with
    ``V10_k(i) = mean_j psi(X_ki, Y_kj)`` over negatives and
    ``V01_k(j) = mean_i psi(X_ki, Y_kj)`` over positives,
    ``S = S10 / m + S01 / n`` where ``S10`` and ``S01`` are the sample
    covariance matrices (ddof=1) of the structural components across the K
    estimates.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    if scores.ndim != 2:
        raise ValueError(f"delong_auc_covariance needs a 2-D (K, n_items) score matrix, got {scores.shape}")
    if scores.shape[1] != labels.size:
        raise ValueError(
            f"score matrix has {scores.shape[1]} items but the label vector has {labels.size}"
        )
    if len(names) != scores.shape[0]:
        raise ValueError(f"got {len(names)} names for {scores.shape[0]} AUC estimates")
    if not np.isfinite(scores).all():
        raise ValueError("delong_auc_covariance received non-finite scores")

    m = int(labels.sum())
    n = int((~labels).sum())
    if m < 2 or n < 2:
        raise ValueError(
            f"DeLong covariance needs at least two items in each label group, got {m} positive / {n} negative"
        )

    k = scores.shape[0]
    aucs = np.empty(k)
    v10 = np.empty((k, m))
    v01 = np.empty((k, n))
    for index in range(k):
        psi = _psi_matrix(scores[index, labels], scores[index, ~labels])
        aucs[index] = psi.mean()
        v10[index] = psi.mean(axis=1)
        v01[index] = psi.mean(axis=0)

    s10 = np.atleast_2d(np.cov(v10, ddof=1))
    s01 = np.atleast_2d(np.cov(v01, ddof=1))
    covariance = s10 / m + s01 / n
    return DeLongResult(
        names=tuple(names),
        aucs=aucs,
        covariance=covariance,
        n_positive=m,
        n_negative=n,
    )


# --------------------------------------------------------------------------
# C4: probability-family minus CDD AUC-rank reversal (§4.5.6)
# --------------------------------------------------------------------------


def is_reversal(gap_baseline: float, gap_target: float) -> bool:
    """§4.5.6's literal reversal rule: ``g_bf16 > 0 and g_nf4 < 0`` or
    ``g_bf16 < 0 and g_nf4 > 0``. A zero gap at either precision is not a
    reversal, and a same-sign change of magnitude is not a reversal."""
    return (gap_baseline > 0 and gap_target < 0) or (gap_baseline < 0 and gap_target > 0)


def _one_sided_p_values(gap: float, se: float) -> tuple[float, float]:
    """``(p_greater, p_less)``: normal Wald one-sided p-values for the
    alternatives ``gap > 0`` and ``gap < 0``."""
    from scipy import stats  # noqa: PLC0415  (see module docstring)

    z = gap / se
    return float(stats.norm.sf(z)), float(stats.norm.cdf(z))


def c4_p_value(
    gap_baseline: float,
    se_baseline: float,
    gap_target: float,
    se_target: float,
) -> dict[str, float]:
    """§4.5.6's intersection-union construction:
    ``p_forward = max(p+_b, p-_q)``, ``p_reverse = max(p-_b, p+_q)``,
    ``p_C4 = min(1, 2 * min(p_forward, p_reverse))``. The factor two corrects
    for choosing either reversal direction."""
    p_plus_b, p_minus_b = _one_sided_p_values(gap_baseline, se_baseline)
    p_plus_q, p_minus_q = _one_sided_p_values(gap_target, se_target)
    p_forward = max(p_plus_b, p_minus_q)
    p_reverse = max(p_minus_b, p_plus_q)
    return {
        "p_forward": p_forward,
        "p_reverse": p_reverse,
        "p_value": min(1.0, 2 * min(p_forward, p_reverse)),
    }


@dataclasses.dataclass(frozen=True)
class C4Result:
    status: str
    p_value: float
    gap_baseline: float | None
    gap_target: float | None
    se_gap_baseline: float | None
    se_gap_target: float | None
    p_forward: float | None
    p_reverse: float | None
    reversal_observed: bool
    component_aucs: dict[str, float]
    n_possible_exposure: int
    n_shared_clean_control: int

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def c4_rank_reversal_test(
    baseline_scores: dict[str, np.ndarray],
    target_scores: dict[str, np.ndarray],
    labels: Sequence[bool] | np.ndarray,
    *,
    baseline: str = "bf16",
    target: str = "bnb_nf4",
) -> C4Result:
    """C4 end to end (§4.5.6).

    `baseline_scores` and `target_scores` each map detector name ->
    per-item score array, all aligned to `labels` (True =
    `possible-exposure`, False = `shared-clean-control`). The six AUCs share
    one DeLong covariance; each gap's SE comes from the corresponding linear
    contrast (weights +1/2, +1/2, -1 within one precision).
    """
    labels = np.asarray(labels, dtype=bool)
    required = (*C4_PROBABILITY_DETECTORS, C4_REFERENCE_DETECTOR)

    def not_estimable(status: str, component_aucs: dict[str, float] | None = None) -> C4Result:
        return C4Result(
            status=status,
            p_value=1.0,
            gap_baseline=None,
            gap_target=None,
            se_gap_baseline=None,
            se_gap_target=None,
            p_forward=None,
            p_reverse=None,
            reversal_observed=False,
            component_aucs=component_aucs or {},
            n_possible_exposure=int(labels.sum()),
            n_shared_clean_control=int((~labels).sum()),
        )

    for detector in required:
        for source in (baseline_scores, target_scores):
            values = source.get(detector)
            if values is None:
                return not_estimable(STATUS_MISSING_SCORES)
            values = np.asarray(values, dtype=float)
            if values.size != labels.size or not np.isfinite(values).all():
                return not_estimable(STATUS_MISSING_SCORES)

    if labels.sum() < 2 or (~labels).sum() < 2:
        return not_estimable(STATUS_EMPTY_LABEL_GROUP)

    names: list[str] = []
    rows: list[np.ndarray] = []
    for precision, source in ((baseline, baseline_scores), (target, target_scores)):
        for detector in required:
            names.append(f"{detector}@{precision}")
            rows.append(np.asarray(source[detector], dtype=float))

    delong = delong_auc_covariance(np.vstack(rows), labels, names=names)
    component_aucs = {name: delong.auc(name) for name in names}

    def gap(precision: str) -> tuple[float, float]:
        weights = {f"{d}@{precision}": 0.5 for d in C4_PROBABILITY_DETECTORS}
        weights[f"{C4_REFERENCE_DETECTOR}@{precision}"] = -1.0
        return delong.contrast(weights)

    gap_b, se_b = gap(baseline)
    gap_q, se_q = gap(target)
    if not (se_b > 0 and se_q > 0) or not (np.isfinite(se_b) and np.isfinite(se_q)):
        return dataclasses.replace(
            not_estimable(STATUS_ZERO_VARIANCE, component_aucs),
            gap_baseline=gap_b,
            gap_target=gap_q,
            se_gap_baseline=se_b if np.isfinite(se_b) else None,
            se_gap_target=se_q if np.isfinite(se_q) else None,
            reversal_observed=is_reversal(gap_b, gap_q),
        )

    p_values = c4_p_value(gap_b, se_b, gap_q, se_q)
    return C4Result(
        status=STATUS_COMPUTED,
        p_value=p_values["p_value"],
        gap_baseline=gap_b,
        gap_target=gap_q,
        se_gap_baseline=se_b,
        se_gap_target=se_q,
        p_forward=p_values["p_forward"],
        p_reverse=p_values["p_reverse"],
        reversal_observed=is_reversal(gap_b, gap_q),
        component_aucs=component_aucs,
        n_possible_exposure=delong.n_positive,
        n_shared_clean_control=delong.n_negative,
    )


# --------------------------------------------------------------------------
# Multiplicity (§4.5.6)
# --------------------------------------------------------------------------


def holm_adjusted_p_values(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, with the usual monotonicity
    enforcement and capping at 1. Rejecting every slot whose adjusted value
    is <= alpha is equivalent to the sequential Holm procedure at familywise
    alpha. Matches `statsmodels.stats.multitest.multipletests(method="holm")`
    (tests/test_confirmatory.py)."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("holm_adjusted_p_values expects a 1-D sequence")
    if p.size == 0:
        return p.astype(float)
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("p-values must lie in [0, 1]")
    k = p.size
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.empty(k)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (k - rank) * float(p[index]))
        adjusted_sorted[rank] = min(1.0, running)
    out = np.empty(k)
    out[order] = adjusted_sorted
    return out


def holm_decisions(p_values: Sequence[float], *, alpha: float = ALPHA) -> np.ndarray:
    """Boolean rejection vector at familywise `alpha`."""
    return holm_adjusted_p_values(p_values) <= alpha


@dataclasses.dataclass(frozen=True)
class ConfirmatoryFamilyResult:
    """The four §4.5.6 slots, their raw p-values, and the Holm decision.
    Not-estimable slots are retained with ``p_value = 1.0``."""

    model: str
    baseline: str
    target: str
    slots: tuple[str, ...]
    tests: dict[str, dict]
    raw_p_values: dict[str, float]
    holm_adjusted_p_values: dict[str, float]
    rejected: dict[str, bool]
    familywise_alpha: float

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def assemble_confirmatory_family(
    q1a_results: dict[str, PairedShiftResult],
    c4_result: C4Result,
    *,
    model: str,
    baseline: str = "bf16",
    target: str = "bnb_nf4",
    familywise_alpha: float = ALPHA,
) -> ConfirmatoryFamilyResult:
    """Combine C1-C3 (keyed by slot name) and C4 into the Holm-corrected
    family of four (§4.5.6). Every slot is kept even when not estimable."""
    missing = set(Q1A_SLOT_DETECTORS) - set(q1a_results)
    if missing:
        raise ValueError(f"missing Q1a slots {sorted(missing)}; §4.5.6 keeps all four slots")
    tests: dict[str, dict] = {slot: q1a_results[slot].as_dict() for slot in ("C1", "C2", "C3")}
    tests["C4"] = c4_result.as_dict()
    raw = [tests[slot]["p_value"] for slot in CONFIRMATORY_SLOTS]
    adjusted = holm_adjusted_p_values(raw)
    return ConfirmatoryFamilyResult(
        model=model,
        baseline=baseline,
        target=target,
        slots=CONFIRMATORY_SLOTS,
        tests=tests,
        raw_p_values=dict(zip(CONFIRMATORY_SLOTS, (float(v) for v in raw))),
        holm_adjusted_p_values=dict(zip(CONFIRMATORY_SLOTS, (float(v) for v in adjusted))),
        rejected={
            slot: bool(value <= familywise_alpha)
            for slot, value in zip(CONFIRMATORY_SLOTS, adjusted)
        },
        familywise_alpha=familywise_alpha,
    )


def run_confirmatory_family(
    q1a_paired_scores: dict[str, tuple[np.ndarray, np.ndarray]],
    c4_baseline_scores: dict[str, np.ndarray],
    c4_target_scores: dict[str, np.ndarray],
    c4_labels: Sequence[bool] | np.ndarray,
    *,
    model: str,
    baseline: str = "bf16",
    target: str = "bnb_nf4",
    familywise_alpha: float = ALPHA,
) -> ConfirmatoryFamilyResult:
    """One call for all four §4.5.6 tests.

    `q1a_paired_scores` maps detector name -> ``(baseline_array,
    target_array)`` over **all** of that model's LCB items (§4.5.6: the 183
    intermediate-date items are included and no exposure label is used).
    The C4 arguments cover only the `possible-exposure` /
    `shared-clean-control` subset.
    """
    q1a_results = {}
    for slot, detector in Q1A_SLOT_DETECTORS.items():
        pair = q1a_paired_scores.get(detector)
        if pair is None:
            q1a_results[slot] = PairedShiftResult(
                detector=detector,
                status=STATUS_MISSING_SCORES,
                n_pairs=0,
                mean_difference=None,
                sd_difference=None,
                standard_error=None,
                t_statistic=None,
                degrees_of_freedom=None,
                p_value=1.0,
                cohens_dz=None,
            )
        else:
            q1a_results[slot] = paired_score_shift_test(pair[0], pair[1], detector=detector)
    c4 = c4_rank_reversal_test(
        c4_baseline_scores, c4_target_scores, c4_labels, baseline=baseline, target=target
    )
    return assemble_confirmatory_family(
        q1a_results,
        c4,
        model=model,
        baseline=baseline,
        target=target,
        familywise_alpha=familywise_alpha,
    )
