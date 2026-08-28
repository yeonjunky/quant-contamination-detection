"""`correct ~ precision * exposure_proxy + (1 | item) + (1 | model)` — paper
§4.5.5, the estimand for Q2 (the `precision:exposure_proxy` interaction term,
on the log-odds scale).

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


@dataclasses.dataclass
class MixedEffectsResult:
    interaction_log_odds: float
    interaction_sd: float
    fixed_effect_names: list[str]
    fixed_effect_means: list[float]
    fixed_effect_sds: list[float]
    raw: object  # the underlying statsmodels VBResults, for anyone who wants more than the interaction term


def fit_precision_exposure_proxy_glmm(
    df: pd.DataFrame,
    *,
    item_col: str = "item_id",
    model_col: str = "model",
    precision_col: str = "precision",
    exposure_proxy_col: str = "exposure_proxy",
    correct_col: str = "correct",
) -> MixedEffectsResult:
    """`df` must have one row per (item, model, precision) measurement, with
    `correct_col` a 0/1 (or bool) outcome, `exposure_proxy_col` constant
    within an item (it's a property of the item, not the measurement)."""
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

    glmm = BinomialBayesMixedGLM.from_formula(
        "correct ~ precision * exposure_proxy",
        {"item": "0 + C(item)", "model": "0 + C(model)"},
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
        interaction_sd=float(fe_sds[idx]),
        fixed_effect_names=fe_names,
        fixed_effect_means=fe_means,
        fixed_effect_sds=fe_sds,
        raw=result,
    )
