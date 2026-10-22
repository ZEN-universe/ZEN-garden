"""Mask for eligible modeled capacity additions."""

import xarray as xr

from zen_garden.elements.technology.masks._utils import year_offset
from zen_garden.model.component_types.mask import GenericMask


class CapacityAdditionEligibleMask(GenericMask):
    """Locations and completion years where modeled additions may be nonzero."""

    name = "capacity_addition_eligible"

    @classmethod
    def build(cls, model_constructor, registry):
        model = model_constructor.optimization_model
        existing = model.parameters.existing_capacities.fillna(0)
        capacity_limit = model.parameters.capacity_limit
        eligible = capacity_limit.notnull() & (capacity_limit > existing)
        if not model_constructor.config.system.allow_investment:
            return xr.zeros_like(eligible, dtype=bool)

        years = list(model.sets["set_years"])
        for technology in model.sets["set_technologies"]:
            delay = year_offset(
                model_constructor, technology, "construction_time", ceil=True
            )
            for completion_year in years:
                investment_year = completion_year - delay
                if investment_year in years:
                    continue
                existing_investment = model.parameters.capacity_investment_existing
                existing_years = existing_investment.coords.get(
                    "set_years_entire_horizon"
                )
                if existing_years is None or investment_year not in existing_years:
                    eligible.loc[technology, :, :, completion_year] = False
                    continue
                committed = (
                    existing_investment.sel(
                        set_technologies=technology,
                        set_years_entire_horizon=investment_year,
                    ).fillna(0)
                    != 0
                )
                eligible.loc[technology, :, :, completion_year] &= committed
        return eligible.fillna(False)
