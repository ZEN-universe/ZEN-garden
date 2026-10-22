"""Mask for capacity potentially supplied by modeled additions."""

import xarray as xr

from zen_garden.elements.technology.masks._utils import active_previous_years
from zen_garden.elements.technology.masks.capacity_addition_eligible_mask import (
    CapacityAdditionEligibleMask,
)
from zen_garden.model.component_types.mask import GenericMask


class PotentialNewCapacityMask(GenericMask):
    """Capacity that may be supplied by a surviving modeled addition."""

    name = "potential_new_capacity"
    dependencies = [CapacityAdditionEligibleMask]

    @classmethod
    def build(cls, model_constructor, registry):
        additions = registry[CapacityAdditionEligibleMask]
        capacity = xr.zeros_like(additions, dtype=bool)
        years = list(model_constructor.optimization_model.sets["set_years"])
        technologies = model_constructor.optimization_model.sets["set_technologies"]
        for technology in technologies:
            for year in years:
                previous_years = active_previous_years(
                    model_constructor, technology, year, "lifetime"
                )
                capacity.loc[technology, :, :, year] = additions.sel(
                    set_technologies=technology,
                    set_years=previous_years,
                ).any("set_years")
        return capacity
