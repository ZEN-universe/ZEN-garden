"""Mask for eligible modeled capacity additions."""

import xarray as xr

from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.technology.masks._utils import year_offset
from zen_garden.elements.technology.parameters.capacity_investment_existing import (
    CapacityInvestmentExisting,
)
from zen_garden.elements.technology.parameters.capacity_limit import CapacityLimit
from zen_garden.elements.technology.parameters.existing_capacities import (
    ExistingCapacities,
)
from zen_garden.model.component_types.mask import GenericMask


class CapacityAdditionEligibleMask(GenericMask):
    """Locations and completion years where modeled additions may be nonzero."""

    name = "capacity_addition_eligible"
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def get_mask(cls, model_constructor, registry):
        model = model_constructor.optimization_model
        existing = model.parameters[ExistingCapacities].fillna(0)
        capacity_limit = model.parameters[CapacityLimit]
        eligible = capacity_limit.notnull() & (capacity_limit > existing)
        if not model_constructor.config.system.allow_investment:
            return xr.zeros_like(eligible, dtype=bool)

        years = list(model.sets[SetYears])
        for technology in model.sets[SetTechnologies]:
            delay = year_offset(
                model_constructor, technology, "construction_time", ceil=True
            )
            for completion_year in years:
                investment_year = completion_year - delay
                if investment_year in years:
                    continue
                existing_investment = model.parameters[CapacityInvestmentExisting]
                existing_years = existing_investment.coords.get(
                    "set_years_entire_horizon"
                )
                if existing_years is None or investment_year not in existing_years:
                    eligible.loc[technology, :, completion_year] = False
                    continue
                committed = (
                    existing_investment.sel(
                        set_technologies=technology,
                        set_years_entire_horizon=investment_year,
                    ).fillna(0)
                    != 0
                )
                eligible.loc[technology, :, completion_year] &= committed
        return eligible.fillna(False)
