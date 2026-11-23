"""Shared computation for the derived ``existing_*`` technology parameters.

``existing_capacities`` and ``existing_capex`` aggregate, per investment year,
the still-active vintages of existing capacity. They are recomputed at every
model-construction step because rolling-horizon runs mutate the underlying
existing-capacity parameters between steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

from zen_garden.elements.energy_system.parameters.interval_between_years import (
    IntervalBetweenYears,
)
from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.technology.parameters.lifetime import Lifetime
from zen_garden.elements.technology.parameters.lifetime_existing import LifetimeExisting
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor

_QUANTITY_INDEX = [
    "set_technologies",
    "set_location",
    "set_years",
]


def compute_existing_quantity(
    model_constructor: "ModelConstructor",
    source_parameter: str,
    *,
    storage_energy: bool = False,
) -> xr.DataArray:
    """Sum ``source_parameter`` over existing-capacity vintages still active.

    :param model_constructor: constructor of the technology being built
    :param source_parameter: name of the per-vintage parameter to aggregate
        (``capacity_existing`` or ``capex_capacity_existing``)
    :return: the aggregated quantity over ``_QUANTITY_INDEX``; positions outside
        the technology index are NaN and dropped on registration
    """
    parameters = model_constructor.optimization_model.parameters
    source = getattr(parameters, source_parameter)
    if source.size == 0:
        quantity = _restrict_to_technology_index(model_constructor, xr.DataArray(0.0))
        return (
            _select_storage_energy_quantity(model_constructor, quantity)
            if storage_energy
            else quantity
        )
    if storage_energy:
        source = source.rename(
            {
                "set_storage_technologies": "set_technologies",
                "set_nodes": "set_location",
            }
        )
    still_active = _still_active_vintages(
        model_constructor, parameters[LifetimeExisting], parameters[Lifetime]
    )
    quantity = (source * still_active).sum("set_existing_capacity_ids")
    quantity = _restrict_to_technology_index(model_constructor, quantity)
    if not storage_energy:
        return quantity
    return _select_storage_energy_quantity(model_constructor, quantity)


def _select_storage_energy_quantity(
    model_constructor: "ModelConstructor", quantity: xr.DataArray
) -> xr.DataArray:
    """Select storage technologies and expose their energy-capacity indices."""
    model = model_constructor.optimization_model
    return quantity.sel(
        set_technologies=model.sets[SetStorageTechnologies],
        set_location=model.sets[SetNodes],
    ).rename(
        {
            "set_technologies": "set_storage_technologies",
            "set_location": "set_nodes",
        }
    )


def _still_active_vintages(
    model_constructor: "ModelConstructor",
    lifetime_existing: xr.DataArray,
    lifetime: xr.DataArray,
) -> xr.DataArray:
    """Boolean mask over (technology, location, vintage, year).

    A vintage is still active in a year if the years elapsed since the start of
    the optimization horizon cover its remaining lifetime overhang relative to a
    newly built unit. NaN remaining lifetimes (absent vintages) resolve to
    ``False``.
    """
    interval = model_constructor.optimization_model.parameters[IntervalBetweenYears]
    years = list(model_constructor.model_schema.set_years)
    year = xr.DataArray(years, coords={"set_years": years}, dims="set_years")
    elapsed = (year - years[0]) * interval

    lifetime_overhang = lifetime_existing - lifetime
    active_when_older = elapsed >= lifetime_overhang
    active_when_newer = elapsed + interval <= lifetime_existing
    return xr.where(lifetime_overhang >= 0, active_when_older, active_when_newer)


def _restrict_to_technology_index(
    model_constructor: "ModelConstructor", quantity: xr.DataArray
) -> xr.DataArray:
    """Keep only valid technology, location, and year tuples."""
    structural_mask = model_constructor.optimization_model.sets.resolve_indices(
        _QUANTITY_INDEX
    ).structural_mask
    return quantity.reindex_like(structural_mask).where(structural_mask)
