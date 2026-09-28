"""Item-stratified conditional logistic fit for β_QE and its Wald interval —
paper §4.5.5's *reported* interval for Q2.

§4.5.5 is explicit that the interval is **not** taken from a mean-field
variational Bayes posterior SD: the mean-field factorization discards the
coefficients' correlations, so in a 2x2 treatment-coded design the posterior
SD is several times smaller than the estimate's own sampling variability and
``mean +- 1.96 SD`` does not carry its nominal coverage. The interval
reported instead comes from an item-stratified conditional logistic fit, run
**separately per model**, in which item difficulty is conditioned out rather
than modelled.

Coding follows §3.1: ``Q = 0`` for bf16, ``Q = 1`` for the quantized level
under test, ``E = 0`` for `shared-clean-control`, ``E = 1`` for
`possible-exposure`. The regressors are ``Q`` and ``Q x E``; ``E`` itself is
constant within an item and so drops out of the conditional likelihood along
with the item intercepts. The reported interval is the Wald interval for
β_QE. §3.1 also fixes ``J = -β_QE``; negating an interval reverses its
endpoints, which `beta_qe_interval_to_j` does explicitly.

§4.5.5 permits two alternatives (MCMC from the full posterior, or a Laplace
approximation inverting the full joint Hessian over fixed effects and random
intercepts together) **only** after the same synthetic-data coverage check is
passed — see `qcd.analysis.coverage`.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd

STATUS_COMPUTED = "computed"
STATUS_NOT_ESTIMABLE = "not_estimable"

#: Default two-sided Wald interval level.
DEFAULT_INTERVAL_LEVEL = 0.95


@dataclasses.dataclass(frozen=True)
class ConditionalLogitResult:
    """β_QE from one model's item-stratified conditional logistic fit.

    `interval_method` is stored on the result so an analysis manifest can
    record which of §4.5.5's permitted methods produced the reported
    interval.
    """

    status: str
    beta_qe: float | None
    standard_error: float | None
    ci_low: float | None
    ci_high: float | None
    beta_q: float | None
    beta_q_standard_error: float | None
    n_items_supplied: int
    n_items_informative: int
    n_observations_used: int
    interval_level: float
    interval_method: str = "item_stratified_conditional_logit_wald"
    message: str | None = None

    @property
    def j(self) -> float | None:
        """§3.1's drop-oriented contrast ``J = -β_QE``."""
        return None if self.beta_qe is None else -self.beta_qe

    @property
    def j_interval(self) -> tuple[float, float] | None:
        """The β_QE interval negated, with endpoints reversed (§4.5.5)."""
        return beta_qe_interval_to_j(self.ci_low, self.ci_high)

    def covers(self, value: float) -> bool | None:
        if self.ci_low is None or self.ci_high is None:
            return None
        return bool(self.ci_low <= value <= self.ci_high)

    def as_dict(self) -> dict:
        payload = dataclasses.asdict(self)
        payload["j"] = self.j
        interval = self.j_interval
        payload["j_ci_low"] = None if interval is None else interval[0]
        payload["j_ci_high"] = None if interval is None else interval[1]
        return payload


def beta_qe_interval_to_j(
    ci_low: float | None, ci_high: float | None
) -> tuple[float, float] | None:
    """``J = -β_QE``, so the interval for J is ``(-high, -low)`` — negating
    an interval also reverses its endpoints (§4.5.5)."""
    if ci_low is None or ci_high is None:
        return None
    return (-ci_high, -ci_low)


def build_qe_design(
    df: pd.DataFrame,
    *,
    baseline: str,
    target: str,
    item_col: str = "item_id",
    precision_col: str = "precision",
    exposure_proxy_col: str = "exposure_proxy",
    correct_col: str = "correct",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn one model's tidy rows into ``(y, X, groups)`` with ``X`` columns
    ``[Q, Q*E]`` under §3.1's coding. Rows at precisions other than
    `baseline`/`target` are dropped."""
    for column in (item_col, precision_col, exposure_proxy_col, correct_col):
        if column not in df.columns:
            raise ValueError(f"conditional-logit input is missing column {column!r}")
    working = df[df[precision_col].isin((baseline, target))].copy()
    if working.empty:
        raise ValueError(f"no rows at precisions {baseline!r}/{target!r}")
    exposure = working[exposure_proxy_col]
    if exposure.map(lambda v: isinstance(v, bool) or v in (0, 1, True, False)).all():
        e = exposure.astype(bool).astype(float).to_numpy()
    else:
        raise ValueError("exposure_proxy must be boolean or 0/1")
    per_item_exposure = working.groupby(item_col)[exposure_proxy_col].nunique()
    if (per_item_exposure > 1).any():
        raise ValueError(
            "exposure_proxy must be constant across precision within an item (§4.5.5)"
        )
    q = working[precision_col].eq(target).astype(float).to_numpy()
    y = working[correct_col].astype(int).to_numpy()
    x = np.column_stack([q, q * e])
    groups = working[item_col].to_numpy()
    return y, x, groups


def fit_conditional_logit_beta_qe(
    df: pd.DataFrame,
    *,
    baseline: str = "bf16",
    target: str = "bnb_nf4",
    item_col: str = "item_id",
    precision_col: str = "precision",
    exposure_proxy_col: str = "exposure_proxy",
    correct_col: str = "correct",
    interval_level: float = DEFAULT_INTERVAL_LEVEL,
) -> ConditionalLogitResult:
    """Fit ``logit Pr(correct) = item_intercept + β_Q Q + β_QE Q*E``
    conditionally on item, for **one model's** rows, and return β_QE with its
    Wald interval (§4.5.5).

    `df` must contain one row per (item, precision) measurement for a single
    model. Items whose outcome is constant across the two precisions carry no
    information in the conditional likelihood and are dropped by the
    conditional fit; `n_items_informative` reports how many remain. A fit
    that cannot be estimated (no informative item, separation, singular
    Hessian) returns ``status = "not_estimable"`` rather than raising.
    """
    from scipy import stats  # noqa: PLC0415
    from statsmodels.discrete.conditional_models import ConditionalLogit  # noqa: PLC0415

    y, x, groups = build_qe_design(
        df,
        baseline=baseline,
        target=target,
        item_col=item_col,
        precision_col=precision_col,
        exposure_proxy_col=exposure_proxy_col,
        correct_col=correct_col,
    )
    n_items_supplied = int(pd.unique(groups).size)
    # statsmodels drops a stratum whose outcome has no within-group variance
    # (conditional_models.py: `if np.std(y) == 0: continue`). Recompute that
    # here so the reported counts do not depend on a private attribute.
    within_item_variance = pd.Series(y).groupby(pd.Series(groups)).nunique()
    informative_items = within_item_variance[within_item_variance > 1]
    n_informative = int(informative_items.size)
    n_used = int(pd.Series(groups).isin(informative_items.index).sum())

    def not_estimable(message: str, informative: int = 0, used: int = 0) -> ConditionalLogitResult:
        return ConditionalLogitResult(
            status=STATUS_NOT_ESTIMABLE,
            beta_qe=None,
            standard_error=None,
            ci_low=None,
            ci_high=None,
            beta_q=None,
            beta_q_standard_error=None,
            n_items_supplied=n_items_supplied,
            n_items_informative=informative,
            n_observations_used=used,
            interval_level=interval_level,
            message=message,
        )

    if x[:, 1].std() == 0 or x[:, 0].std() == 0:
        return not_estimable("Q or Q*E is constant; β_QE is not identified")
    if n_informative == 0:
        return not_estimable("no item has a within-item outcome difference across precisions")
    # Only the retained (discordant) strata enter the conditional likelihood,
    # so Q*E must still vary among *those* rows. If, say, every
    # possible-exposure item is concordant, β_QE is flat in the likelihood.
    retained = pd.Series(groups).isin(informative_items.index).to_numpy()
    if x[retained, 1].std() == 0:
        return not_estimable(
            "Q*E is constant among the items that contribute to the conditional "
            "likelihood; β_QE is not identified",
            n_informative,
            n_used,
        )

    try:
        with warnings.catch_warnings():
            # statsmodels warns about dropping concordant strata; that is the
            # defining behaviour of a conditional likelihood, not a problem.
            warnings.simplefilter("ignore")
            model = ConditionalLogit(y, x, groups=groups)
            result = model.fit(disp=0)
    except Exception as exc:  # noqa: BLE001 - reported, never silently swallowed
        return not_estimable(
            f"conditional logistic fit failed: {type(exc).__name__}: {exc}",
            n_informative,
            n_used,
        )

    params = np.asarray(result.params, dtype=float)
    bse = np.asarray(result.bse, dtype=float)
    if params.size != 2 or not np.isfinite(params).all():
        return not_estimable("conditional logistic fit returned non-finite coefficients", n_informative, n_used)
    if not np.isfinite(bse).all() or bse[1] <= 0:
        return not_estimable(
            "conditional logistic Wald standard error is zero or undefined", n_informative, n_used
        )

    z = float(stats.norm.ppf(0.5 + interval_level / 2))
    beta_qe = float(params[1])
    se = float(bse[1])
    return ConditionalLogitResult(
        status=STATUS_COMPUTED,
        beta_qe=beta_qe,
        standard_error=se,
        ci_low=beta_qe - z * se,
        ci_high=beta_qe + z * se,
        beta_q=float(params[0]),
        beta_q_standard_error=float(bse[0]),
        n_items_supplied=n_items_supplied,
        n_items_informative=n_informative,
        n_observations_used=n_used,
        interval_level=interval_level,
    )


def fit_conditional_logit_by_model(
    df: pd.DataFrame,
    *,
    model_col: str = "model",
    **kwargs,
) -> dict[str, ConditionalLogitResult]:
    """§4.5.5's "run separately per model" — one independent item-stratified
    fit per model, never a pooled fit with a model term."""
    return {
        str(model): fit_conditional_logit_beta_qe(cell, **kwargs)
        for model, cell in df.groupby(model_col)
    }
