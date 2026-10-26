import xarray as xr

from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.storage_technology.masks.energy_capacity_addition_eligible_mask import (  # noqa: E501
    EnergyCapacityAdditionEligibleMask,
)
from zen_garden.elements.technology.masks._utils import active_previous_years
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.mask import GenericMask


class EnergyModeledCapexMask(GenericMask):
    """Modeled storage-energy CAPEX within its depreciation period."""

    name = "energy_modeled_capex"
    indices = ("set_technologies", "set_location", "set_years")
    dependencies = (EnergyCapacityAdditionEligibleMask,)

    @classmethod
    def get_mask(cls, model_constructor, registry):
        additions = registry[EnergyCapacityAdditionEligibleMask]
        capex = xr.zeros_like(additions, dtype=bool)
        years = list(model_constructor.optimization_model.sets[SetYears])
        technologies = model_constructor.optimization_model.sets[SetStorageTechnologies]
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
