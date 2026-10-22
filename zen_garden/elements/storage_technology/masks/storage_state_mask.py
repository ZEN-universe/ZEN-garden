"""Mask for storage state variables and constraints."""

import xarray as xr

from zen_garden.elements.storage_technology.masks.storage_inflow_mask import (
    StorageInflowMask,
)
from zen_garden.elements.technology.masks import CapacityAnyTypeMask
from zen_garden.model.component_types.mask import GenericMask


class StorageStateMask(GenericMask):
    """Storage states required for a year containing capacity or inflow."""

    name = "storage_state"
    dependencies = [CapacityAnyTypeMask, StorageInflowMask]

    @classmethod
    def build(cls, model_constructor, registry):
        model = model_constructor.optimization_model
        available = registry[CapacityAnyTypeMask]
        inflow = registry[StorageInflowMask]
        storage_technologies = list(model.sets["set_storage_technologies"])
        nodes = list(model.sets["set_nodes"])
        storage_times = list(model.sets["set_time_steps_storage"])
        mask = xr.DataArray(
            False,
            coords={
                "set_technologies": storage_technologies,
                "set_location": nodes,
                "set_time_steps_storage": storage_times,
            },
            dims=(
                "set_technologies",
                "set_location",
                "set_time_steps_storage",
            ),
        )
        for year in model.sets["set_years"]:
            operation_times = (
                model_constructor.time_steps.get_time_steps_year2operation(year)
            )
            year_active = available.sel(
                set_technologies=storage_technologies,
                set_location=nodes,
                set_years=year,
            ) | inflow.sel(
                set_technologies=storage_technologies,
                set_location=nodes,
                set_time_steps_operation=operation_times,
            ).any(
                "set_time_steps_operation"
            )
            mask.loc[
                :, :, model_constructor.time_steps.get_time_steps_year2storage(year)
            ] = year_active
        return mask
