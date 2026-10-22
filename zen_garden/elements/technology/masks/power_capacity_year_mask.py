"""Power-capacity availability mask by year."""

from zen_garden.elements.technology.masks.capacity_available_mask import (
    CapacityAvailableMask,
)
from zen_garden.model.component_types.mask import GenericMask


class PowerCapacityYearMask(GenericMask):
    """Power-capacity availability without the capacity-type dimension."""

    name = "power_capacity_year"
    dependencies = [CapacityAvailableMask]

    @classmethod
    def build(cls, model_constructor, registry):
        mask = registry[CapacityAvailableMask].sel(set_capacity_types="power")
        return mask.drop_vars("set_capacity_types", errors="ignore")
