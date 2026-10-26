"""Power-capacity addition mask."""

from zen_garden.elements.technology.masks.capacity_addition_eligible_mask import (
    CapacityAdditionEligibleMask,
)
from zen_garden.model.component_types.mask import GenericMask


class PowerCapacityAdditionMask(GenericMask):
    """Eligible power-capacity additions without the capacity-type dimension."""

    name = "power_capacity_addition"
    indices = ("set_technologies", "set_location", "set_years")
    dependencies = (CapacityAdditionEligibleMask,)

    @classmethod
    def get_mask(cls, model_constructor, registry):
        return registry[CapacityAdditionEligibleMask]
