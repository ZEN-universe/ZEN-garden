"""Capacity-addition mask reduced across capacity types."""

from zen_garden.elements.technology.masks.capacity_addition_eligible_mask import (
    CapacityAdditionEligibleMask,
)
from zen_garden.model.component_types.mask import GenericMask


class CapacityAdditionAnyTypeMask(GenericMask):
    """Addition eligibility reduced across power and energy capacity types."""

    name = "capacity_addition_any_type"
    dependencies = [CapacityAdditionEligibleMask]

    @classmethod
    def build(cls, model_constructor, registry):
        return registry[CapacityAdditionEligibleMask].any("set_capacity_types")
