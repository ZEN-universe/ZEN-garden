"""Capacity-availability mask reduced across capacity types."""

from zen_garden.elements.technology.masks.capacity_available_mask import (
    CapacityAvailableMask,
)
from zen_garden.model.component_types.mask import GenericMask


class CapacityAnyTypeMask(GenericMask):
    """Capacity availability reduced across power and energy capacity types."""

    name = "capacity_any_type"
    dependencies = [CapacityAvailableMask]

    @classmethod
    def build(cls, model_constructor, registry):
        return registry[CapacityAvailableMask].any("set_capacity_types")
