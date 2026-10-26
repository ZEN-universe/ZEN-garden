"""Mask for CAPEX associated with modeled additions."""

import xarray as xr

from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.technology.masks._utils import active_previous_years
from zen_garden.elements.technology.masks.capacity_addition_eligible_mask import (
    CapacityAdditionEligibleMask,
)
from zen_garden.model.component_types.mask import GenericMask


class ModeledCapexMask(GenericMask):
    """Modeled overnight CAPEX still inside its depreciation period."""

    name = "modeled_capex"
    indices = ("set_technologies", "set_location", "set_years")
    dependencies = (CapacityAdditionEligibleMask,)

    @classmethod
    def get_mask(cls, model_constructor, registry):
        additions = registry[CapacityAdditionEligibleMask]
        capex = xr.zeros_like(additions, dtype=bool)
        years = list(model_constructor.optimization_model.sets[SetYears])
        technologies = model_constructor.optimization_model.sets[SetTechnologies]
        for technology in technologies:
            for year in years:
                previous_years = active_previous_years(
                    model_constructor, technology, year, "depreciation_time"
                )
                capex.loc[technology, :, year] = additions.sel(
                    set_technologies=technology,
                    set_years=previous_years,
                ).any("set_years")
        return capex
