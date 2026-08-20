"""Known-answer test for analysis/mixed_effects.py — simulate data from a
*known* true precision:contaminated interaction coefficient and confirm the
fitted GLMM recovers it within a wide tolerance. Unlike the closed-form
power/AUC/logodds tables elsewhere in this suite, a variational-Bayes GLMM
fit isn't exactly reproducible to many decimals against a hand-derived
target — there is no citation number to match digit-for-digit — so this is
a recovery-*range* check (does the fit land in the right neighborhood, with
the right sign), not a tight numerical regression.
"""

import numpy as np
import pandas as pd
import pytest

from qcd.analysis.mixed_effects import fit_precision_contamination_glmm


def _simulate(
    *,
    n_items: int,
    n_models: int,
    true_intercept: float,
    true_precision: float,
    true_contaminated: float,
    true_interaction: float,
    item_sd: float = 1.0,
    model_sd: float = 0.3,
    contaminated_rate: float = 0.4,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    item_effect = rng.normal(0, item_sd, n_items)
    model_effect = rng.normal(0, model_sd, n_models)
    contaminated = rng.rand(n_items) < contaminated_rate

    rows = []
    for i in range(n_items):
        for m in range(n_models):
            for precision in ("bf16", "quant"):
                is_quant = precision == "quant"
                logit = (
                    true_intercept
                    + item_effect[i]
                    + model_effect[m]
                    + true_precision * is_quant
                    + true_contaminated * contaminated[i]
                    + true_interaction * is_quant * contaminated[i]
                )
                p = 1 / (1 + np.exp(-logit))
                rows.append(
                    {
                        "item_id": f"item{i}",
                        "model": f"model{m}",
                        "precision": precision,
                        "contaminated": bool(contaminated[i]),
                        "correct": int(rng.rand() < p),
                    }
                )
    return pd.DataFrame(rows)


def test_glmm_recovers_interaction_sign_and_rough_magnitude():
    true_interaction = 0.8
    df = _simulate(
        n_items=200, n_models=3,
        true_intercept=0.4, true_precision=-0.5, true_contaminated=0.2,
        true_interaction=true_interaction, seed=42,
    )

    result = fit_precision_contamination_glmm(df)

    assert result.interaction_log_odds == pytest.approx(true_interaction, abs=0.5)
    assert result.interaction_sd > 0
    assert len(result.fixed_effect_names) == len(result.fixed_effect_means) == len(result.fixed_effect_sds) == 4
    assert any(":" in name for name in result.fixed_effect_names)


def test_glmm_recovers_near_zero_interaction():
    df = _simulate(
        n_items=200, n_models=3,
        true_intercept=0.4, true_precision=-0.5, true_contaminated=0.2,
        true_interaction=0.0, seed=7,
    )

    result = fit_precision_contamination_glmm(df)

    assert abs(result.interaction_log_odds) < 0.5
