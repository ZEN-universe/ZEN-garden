"""Mask for eligible modeled capacity investments."""

import xarray as xr

from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.technology.masks._utils import year_offset
from zen_garden.elements.technology.masks.capacity_addition_eligible_mask import (
    CapacityAdditionEligibleMask,
)
from zen_garden.model.component_types.mask import GenericMask


class InvestmentEligibleMask(GenericMask):
    """Investment years that map to an eligible completion inside the horizon."""

    name = "investment_eligible"
    indices = ("set_technologies", "set_location", "set_years")
    dependencies = (CapacityAdditionEligibleMask,)

    @classmethod
    def get_mask(cls, model_constructor, registry):
        additions = registry[CapacityAdditionEligibleMask]
        investments = xr.zeros_like(additions, dtype=bool)
        years = list(model_constructor.optimization_model.sets[SetYears])
        technologies = model_constructor.optimization_model.sets[SetTechnologies]
        for technology in technologies:
            delay = year_offset(
                model_constructor, technology, "construction_time", ceil=True
            )
            for completion_year in years:
                investment_year = completion_year - delay
                if investment_year in years:
                    investments.loc[technology, :, investment_year] |= additions.loc[
                        technology, :, completion_year
                    ]
        return investments
