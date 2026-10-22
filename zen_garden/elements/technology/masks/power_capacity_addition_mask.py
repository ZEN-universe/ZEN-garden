"""Power-capacity addition mask."""

from zen_garden.elements.technology.masks.capacity_addition_eligible_mask import (
    CapacityAdditionEligibleMask,
)
from zen_garden.model.component_types.mask import GenericMask


class PowerCapacityAdditionMask(GenericMask):
    """Eligible power-capacity additions without the capacity-type dimension."""

    name = "power_capacity_addition"
    dependencies = [CapacityAdditionEligibleMask]

    @classmethod
    def build(cls, model_constructor, registry):
        mask = registry[CapacityAdditionEligibleMask].sel(set_capacity_types="power")
        return mask.drop_vars("set_capacity_types", errors="ignore")
