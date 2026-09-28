"""Materialize paper §4.2's temporal proxy labels per (model, item)."""

from __future__ import annotations

import datetime as dt
from typing import Iterable

from qcd.config import ModelSpec
from qcd.data.schema import Dataset, Item, TemporalProxyLabel


def _date(value: str | dt.datetime) -> dt.datetime:
    return value if isinstance(value, dt.datetime) else dt.datetime.fromisoformat(value)


def _label_lcb_date(
    contest_date: dt.datetime,
    model_boundary: dt.datetime,
    shared_control_boundary: dt.datetime,
) -> TemporalProxyLabel:
    if contest_date >= shared_control_boundary:
        return TemporalProxyLabel.SHARED_CLEAN_CONTROL
    if contest_date < model_boundary:
        return TemporalProxyLabel.POSSIBLE_EXPOSURE
    return TemporalProxyLabel.CLEAN_BY_MODEL_CUTOFF


def materialize_model_item_labels(
    items: Iterable[Item],
    models: Iterable[ModelSpec],
    *,
    shared_control_boundary: dt.datetime,
) -> list[dict]:
    """Return one auditable row per (model, item).

    HumanEval and MBPP+ are possible-exposure proxy conditions for every
    retained model. LCB labels are derived from each model's own first-post
    boundary; the shared control overrides model-local clean metadata only on
    or after the latest primary boundary.
    """
    rows: list[dict] = []
    for model in models:
        if model.primary_first_post_boundary is None:
            raise ValueError(f"model {model.name} has no primary temporal boundary")
        primary_boundary = _date(model.primary_first_post_boundary)
        if shared_control_boundary < primary_boundary:
            raise ValueError(
                f"shared control boundary {shared_control_boundary.date()} precedes "
                f"{model.name}'s primary boundary {primary_boundary.date()}"
            )
        sensitivity_boundary = (
            _date(model.sensitivity_first_post_boundary)
            if model.sensitivity_first_post_boundary is not None
            else primary_boundary
        )
        for item in items:
            publication_date: str | None = None
            if item.dataset in (Dataset.HUMANEVAL, Dataset.MBPPPLUS):
                primary = sensitivity = TemporalProxyLabel.POSSIBLE_EXPOSURE
                if item.metadata.get("publication_date") is not None:
                    publication_date = _date(
                        str(item.metadata["publication_date"])
                    ).date().isoformat()
            else:
                try:
                    contest_date = _date(str(item.metadata["contest_date"]))
                except KeyError:
                    raise ValueError(
                        f"LCB item {item.item_id!r} has no contest_date metadata"
                    ) from None
                publication_date = contest_date.date().isoformat()
                primary = _label_lcb_date(
                    contest_date, primary_boundary, shared_control_boundary
                )
                sensitivity = _label_lcb_date(
                    contest_date, sensitivity_boundary, shared_control_boundary
                )
            rows.append(
                {
                    "model": model.name,
                    "item_id": item.item_id,
                    "dataset": item.dataset.value,
                    "publication_date": publication_date,
                    "primary_label": primary.value,
                    "sensitivity_label": sensitivity.value,
                    "boundary_ambiguous": primary is not sensitivity,
                    "primary_first_post_date": primary_boundary.date().isoformat(),
                    "sensitivity_first_post_date": (
                        sensitivity_boundary.date().isoformat()
                        if model.sensitivity_first_post_boundary is not None
                        else None
                    ),
                    "shared_control_start_date": shared_control_boundary.date().isoformat(),
                }
            )
    return rows
