"""Storage-time mappings used by storage balance expressions."""

from typing import TYPE_CHECKING, cast

import xarray as xr

from zen_garden.elements.energy_system.sets.set_representative_hours_storage import (
    SetRepresentativeHoursStorage,
)
from zen_garden.elements.energy_system.sets.set_years import SetYears

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor


def get_previous_storage_time_step_arrays(
    model_constructor: "ModelConstructor",
):
    """Return predecessor year/hour arrays and the valid-coupling mask."""
    optimization_model = model_constructor.optimization_model
    time_steps = model_constructor.time_steps
    previous = time_steps.previous_storage_time_step
    coupling = time_steps.storage_coupling_mask
    assert previous is not None
    assert coupling is not None
    years = list(optimization_model.sets[SetYears])
    hours = list(
        optimization_model.sets[SetRepresentativeHoursStorage].coordinate_values
    )
    previous_year = xr.DataArray(
        [[year for _ in hours] for year in years],
        coords={"set_years": years, "set_representative_hours_storage": hours},
        dims=("set_years", "set_representative_hours_storage"),
    )
    previous_hour = xr.DataArray(
        [hours for _ in years], coords=previous_year.coords, dims=previous_year.dims
    )
    mask = xr.full_like(previous_year, False, dtype=bool)
    selected_years = set(years)
    previous_steps = cast(dict[tuple[int, int], tuple[int, int]], previous.to_dict())
    coupling_values = cast(dict[tuple[int, int], bool], coupling.to_dict())
    for pair, predecessor in previous_steps.items():
        year, storage_hour = pair
        if year not in selected_years:
            continue
        predecessor_year, predecessor_hour = predecessor
        if predecessor_year not in selected_years:
            valid_hours = time_steps.representative_hours_storage
            assert valid_hours is not None
            predecessor_year = year
            predecessor_hour = valid_hours[year][-1]
        previous_year.loc[year, storage_hour] = predecessor_year
        previous_hour.loc[year, storage_hour] = predecessor_hour
        mask.loc[year, storage_hour] = coupling_values[pair]
    return previous_year, previous_hour, mask
