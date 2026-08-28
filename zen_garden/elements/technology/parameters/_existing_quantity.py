"""Shared computation for the derived ``existing_*`` technology parameters.

``existing_capacities`` and ``existing_capex`` aggregate, per investment year,
the still-active vintages of existing capacity. They are recomputed at every
model-construction step because rolling-horizon runs mutate the underlying
existing-capacity parameters between steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from zen_garden.elements.model_constructor import ModelConstructor

_QUANTITY_INDEX = [
    "set_technologies",
    "set_capacity_types",
    "set_location",
    "set_years",
]


def compute_existing_quantity(
    model_constructor: "ModelConstructor", source_parameter: str
) -> xr.DataArray:
    """Sum ``source_parameter`` over existing-capacity vintages still active.

    :param model_constructor: constructor of the technology being built
    :param source_parameter: name of the per-vintage parameter to aggregate
        (``capacity_existing`` or ``capex_capacity_existing``)
    :return: the aggregated quantity over ``_QUANTITY_INDEX``; positions outside
        the technology index are NaN and dropped on registration
    """
    parameters = model_constructor.zen_model.parameters
    source = getattr(parameters, source_parameter)
    still_active = _still_active_vintages(
        model_constructor, parameters.lifetime_existing, parameters.lifetime
    )
    quantity = (source * still_active).sum("set_technologies_existing")
    return _restrict_to_technology_index(model_constructor, quantity)


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
    interval = model_constructor.config.system.interval_between_years
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
    """Keep only (technology, capacity_type, location, year) tuples in the index."""
    index_values, index_names = model_constructor.create_custom_set(_QUANTITY_INDEX)
    index_arrays = model_constructor.zen_model.sets.tuple_to_arr(
        index_values, index_names
    )
    coords = [
        model_constructor.zen_model.sets.get_coord(values, name)
        for values, name in zip(index_arrays, index_names, strict=False)
    ]
    in_index = xr.DataArray(False, coords=coords, dims=index_names)
    in_index.loc[index_arrays] = True
    return quantity.reindex_like(in_index).where(in_index)
