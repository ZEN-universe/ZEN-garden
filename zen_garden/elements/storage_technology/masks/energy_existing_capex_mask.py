from zen_garden.elements.storage_technology.masks._utils import canonical_storage_array
from zen_garden.elements.storage_technology.parameters.existing_capex_energy import (
    ExistingCapexEnergy,
)
from zen_garden.model.component_types.mask import GenericMask


class EnergyExistingCapexMask(GenericMask):
    """Outstanding CAPEX from existing storage energy capacity."""

    name = "energy_existing_capex"
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def get_mask(cls, model_constructor, registry):
        existing = model_constructor.optimization_model.parameters[ExistingCapexEnergy]
        return canonical_storage_array(existing.fillna(0) != 0)
