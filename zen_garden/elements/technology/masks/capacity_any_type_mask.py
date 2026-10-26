"""Capacity-availability mask reduced across capacity types."""

from zen_garden.elements.technology.masks.capacity_available_mask import (
    CapacityAvailableMask,
)
from zen_garden.model.component_types.mask import GenericMask


class CapacityAnyTypeMask(GenericMask):
    """Capacity availability reduced across power and energy capacity types."""

    name = "capacity_any_type"
    indices = ("set_technologies", "set_location", "set_years")
    dependencies = (CapacityAvailableMask,)

    @classmethod
    def get_mask(cls, model_constructor, registry):
        return registry[CapacityAvailableMask]
