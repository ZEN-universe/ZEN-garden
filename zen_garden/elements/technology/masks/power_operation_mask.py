"""Operational mask derived from available power capacity."""

import xarray as xr

from zen_garden.elements.technology.masks.capacity_available_mask import (
    CapacityAvailableMask,
)
from zen_garden.model.component_types.mask import GenericMask


class PowerOperationMask(GenericMask):
    """Technology power capacity available in each operational time step."""

    name = "power_operation"
    dependencies = [CapacityAvailableMask]

    @classmethod
    def build(cls, model_constructor, registry):
        capacity = registry[CapacityAvailableMask].sel(set_capacity_types="power")
        capacity = capacity.drop_vars("set_capacity_types", errors="ignore")
        times = model_constructor.optimization_model.sets["set_time_steps_operation"]
        years = xr.DataArray(
            [
                model_constructor.time_steps.convert_time_step_operation2year(time)
                for time in times
            ],
            coords={"set_time_steps_operation": times},
            dims="set_time_steps_operation",
        )
        return capacity.sel(set_years=years).drop_vars("set_years", errors="ignore")
