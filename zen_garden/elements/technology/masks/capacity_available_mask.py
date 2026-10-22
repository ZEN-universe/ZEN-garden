"""Mask for capacity available by model year."""

from zen_garden.elements.technology.masks.existing_capacity_mask import (
    ExistingCapacityMask,
)
from zen_garden.elements.technology.masks.potential_new_capacity_mask import (
    PotentialNewCapacityMask,
)
from zen_garden.model.component_types.mask import GenericMask


class CapacityAvailableMask(GenericMask):
    """Existing or potentially constructed capacity available by model year."""

    name = "capacity_available"
    dependencies = [ExistingCapacityMask, PotentialNewCapacityMask]

    @classmethod
    def build(cls, model_constructor, registry):
        return registry[ExistingCapacityMask] | registry[PotentialNewCapacityMask]
