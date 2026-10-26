from zen_garden.elements.storage_technology.masks.energy_existing_capacity_mask import (
    EnergyExistingCapacityMask,
)
from zen_garden.elements.storage_technology.masks.energy_potential_new_capacity_mask import (  # noqa: E501
    EnergyPotentialNewCapacityMask,
)
from zen_garden.model.component_types.mask import GenericMask


class EnergyCapacityAvailableMask(GenericMask):
    """Existing or potentially constructed storage energy capacity."""

    name = "energy_capacity_available"
    indices = ("set_technologies", "set_location", "set_years")
    dependencies = (EnergyExistingCapacityMask, EnergyPotentialNewCapacityMask)

    @classmethod
    def get_mask(cls, model_constructor, registry):
        return (
            registry[EnergyExistingCapacityMask]
            | registry[EnergyPotentialNewCapacityMask]
        )
