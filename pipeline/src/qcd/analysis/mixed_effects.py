"""`correct ~ precision * exposure_proxy + (1 | item) + (1 | model)` — paper
§4.5.5's exploratory pooled working model. This module supplies the **point
estimate and variance components only**.

**It does not supply an interval.** §4.5.5: "The interval for β_QE is not
taken from a mean-field variational Bayes posterior SD." A mean-field
approximation factorizes the posterior across coefficients and discards
their correlations, so in a 2x2 treatment-coded design the posterior SD is
several times smaller than the estimate's own sampling variability and
``mean ± 1.96 SD`` does not carry its nominal coverage. §4.5.5 permits
`BinomialBayesMixedGLM.fit_vb` for point estimates and variance components,
"but not for interval width". The reported β_QE interval comes from
`qcd.analysis.conditional_logit.fit_conditional_logit_beta_qe`, whose
achieved coverage is checked on synthetic data by `qcd.analysis.coverage`.
`MixedEffectsResult.interaction_interval()` therefore raises rather than
returning anything.

**Per-model fits omit the model random intercept** (§4.5.5, final sentence of
the model paragraph). `fit_precision_exposure_proxy_glmm` detects a
single-model frame and fits `(1 | item)` alone; `include_model_random_effect`
can force either behaviour explicitly. A single-model frame with a model
random intercept is not fitted at all — one level carries no variance
component and the term is not identified.

**Dependency note (a deliberate, flagged deviation from CLAUDE.md §6's
"scipy 없음"):** that rule is scoped — by its own text and by how
analysis/_stats.py's docstring invokes it — to *hand-reproducing the
paper's own worked power/AUC tables*, where the failure mode being guarded
against is silently re-deriving a citation number by an inconsistent path.
Fitting a crossed-random-effects logistic GLMM on real, post-hoc
experimental data is a different kind of computation: there is no
hand-verifiable target number to match it against, and reimplementing a
GLMM optimizer from scratch via bisection would itself be exactly the kind
of unverified numerical code that rule's broader discipline warns against.
This module uses `statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM`
(variational Bayes fit) — the standard Python-native crossed-effects
logistic GLMM — added to requirements-local.txt for this module alone.

The interaction term's exact fitted-parameter *name* is data-dependent
(patsy's category-reference-level choice, e.g. "precision[T.quant]:
exposure_proxy[T.True]" vs some other level ordering) so it's located
programmatically (any fixed-effect name containing ":"), never assumed.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

#: Recorded on every result so an analysis manifest states, in the artifact
#: itself, that this fit is not the interval source (§4.5.5).
INTERVAL_SOURCE = "not_an_interval_source__see_qcd.analysis.conditional_logit"


class VariationalIntervalNotPermitted(Exception):
    """Raised when a caller asks this module for a β_QE interval. §4.5.5
    forbids reporting `mean ± 1.96 × mean-field posterior SD` as an
    interval; use `qcd.analysis.conditional_logit` instead."""


@dataclasses.dataclass
class MixedEffectsResult:
    """Point estimate and variance components from the §4.5.5 working model.

    `interaction_posterior_sd` is a mean-field **variational posterior SD**,
    not a frequentist standard error and not half an interval width. It is
    kept for variance-component reporting and named so that no downstream
    JSON key can read as an interval.
    """

    interaction_log_odds: float
    interaction_posterior_sd: float
    fixed_effect_names: list[str]
    fixed_effect_means: list[float]
    fixed_effect_posterior_sds: list[float]
    random_effect_groups: tuple[str, ...]
    interval_source: str = INTERVAL_SOURCE
    raw: object = None  # the underlying statsmodels VBResults, for anyone who wants more

    def interaction_interval(self) -> tuple[float, float]:
        raise VariationalIntervalNotPermitted(
            "paper §4.5.5 forbids taking the β_QE interval from a mean-field "
            "variational Bayes posterior SD; the reported interval is the Wald "
            "interval from qcd.analysis.conditional_logit."
            "fit_conditional_logit_beta_qe, whose coverage is checked by "
            "qcd.analysis.coverage."
        )

    def as_dict(self) -> dict:
        """JSON-ready payload. The posterior-SD key is spelled out as not an
        interval so a reader of the artifact cannot mistake it for one."""
        return {
            "interaction_log_odds": self.interaction_log_odds,
            "interaction_posterior_sd_not_an_interval": self.interaction_posterior_sd,
            "fixed_effect_names": list(self.fixed_effect_names),
            "fixed_effect_means": list(self.fixed_effect_means),
            "fixed_effect_posterior_sds_not_intervals": list(self.fixed_effect_posterior_sds),
            "random_effect_groups": list(self.random_effect_groups),
            "interval_source": self.interval_source,
        }


def fit_precision_exposure_proxy_glmm(
    df: pd.DataFrame,
    *,
    item_col: str = "item_id",
    model_col: str = "model",
    precision_col: str = "precision",
    exposure_proxy_col: str = "exposure_proxy",
    correct_col: str = "correct",
    include_model_random_effect: bool | None = None,
) -> MixedEffectsResult:
    """`df` must have one row per (item, model, precision) measurement, with
    `correct_col` a 0/1 (or bool) outcome, `exposure_proxy_col` constant
    across precision within each model-item pair (it may differ by model).
    Supply one quantized level and bf16, with bf16 as reference and False
    as the exposure reference. The returned interaction is the quantized
    minus bf16 coefficient, i.e. β_QE, the negative of the paper's drop
    contrast J.

    `include_model_random_effect` defaults to "only when more than one model
    is present": §4.5.5's per-model fits omit `(1 | model)`. Asking for it
    on a single-model frame raises.

    The returned SDs are approximate variational posterior SDs, not
    frequentist SEs, and §4.5.5 forbids using them as an interval.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM  # noqa: PLC0415

    working = df.rename(
        columns={
            item_col: "item",
            model_col: "model",
            precision_col: "precision",
            exposure_proxy_col: "exposure_proxy",
            correct_col: "correct",
        }
    )
    n_models = int(working["model"].nunique())
    if include_model_random_effect is None:
        include_model_random_effect = n_models > 1
    if include_model_random_effect and n_models < 2:
        raise ValueError(
            "a model random intercept needs at least two models; §4.5.5's "
            "per-model fits omit (1 | model)"
        )

    variance_components = {"item": "0 + C(item)"}
    if include_model_random_effect:
        variance_components["model"] = "0 + C(model)"

    glmm = BinomialBayesMixedGLM.from_formula(
        "correct ~ precision * exposure_proxy",
        variance_components,
        working,
    )
    result = glmm.fit_vb()

    fe_names = list(glmm.fep_names)
    fe_means = list(result.fe_mean)
    fe_sds = list(result.fe_sd)

    interaction_indices = [i for i, name in enumerate(fe_names) if ":" in name]
    if len(interaction_indices) != 1:
        raise RuntimeError(
            f"expected exactly one precision:exposure_proxy interaction term in the "
            f"fitted fixed effects, found {len(interaction_indices)} (names: {fe_names})"
        )
    idx = interaction_indices[0]

    return MixedEffectsResult(
        interaction_log_odds=float(fe_means[idx]),
        interaction_posterior_sd=float(fe_sds[idx]),
        fixed_effect_names=fe_names,
        fixed_effect_means=fe_means,
        fixed_effect_posterior_sds=fe_sds,
        random_effect_groups=tuple(variance_components),
        raw=result,
    )
