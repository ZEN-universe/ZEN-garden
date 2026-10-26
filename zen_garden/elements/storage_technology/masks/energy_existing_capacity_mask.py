from zen_garden.elements.storage_technology.masks._utils import canonical_storage_array
from zen_garden.elements.storage_technology.parameters.existing_capacities_energy import (  # noqa: E501
    ExistingCapacitiesEnergy,
)
from zen_garden.model.component_types.mask import GenericMask


class EnergyExistingCapacityMask(GenericMask):
    """Existing storage energy capacity available by node and year."""

    name = "energy_existing_capacity"
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def get_mask(cls, model_constructor, registry):
        existing = model_constructor.optimization_model.parameters[
            ExistingCapacitiesEnergy
        ]
        return canonical_storage_array(existing.fillna(0) > 0)
