from zen_garden.elements.storage_technology.masks.energy_existing_capex_mask import (
    EnergyExistingCapexMask,
)
from zen_garden.elements.storage_technology.masks.energy_modeled_capex_mask import (
    EnergyModeledCapexMask,
)
from zen_garden.model.component_types.mask import GenericMask


class EnergyCapexActiveMask(GenericMask):
    """Existing or modeled storage-energy CAPEX active in a modeled year."""

    name = "energy_capex_active"
    indices = ("set_technologies", "set_location", "set_years")
    dependencies = (EnergyExistingCapexMask, EnergyModeledCapexMask)

    @classmethod
    def get_mask(cls, model_constructor, registry):
        return registry[EnergyExistingCapexMask] | registry[EnergyModeledCapexMask]
