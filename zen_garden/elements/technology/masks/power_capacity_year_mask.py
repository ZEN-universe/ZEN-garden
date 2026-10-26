"""Power-capacity availability mask by year."""

from zen_garden.elements.technology.masks.capacity_available_mask import (
    CapacityAvailableMask,
)
from zen_garden.model.component_types.mask import GenericMask


class PowerCapacityYearMask(GenericMask):
    """Power-capacity availability without the capacity-type dimension."""

    name = "power_capacity_year"
    indices = ("set_technologies", "set_location", "set_years")
    dependencies = (CapacityAvailableMask,)

    @classmethod
    def get_mask(cls, model_constructor, registry):
        return registry[CapacityAvailableMask]
