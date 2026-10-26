"""Mask for capacity committed outside the optimization decisions."""

from zen_garden.elements.technology.parameters.existing_capacities import (
    ExistingCapacities,
)
from zen_garden.model.component_types.mask import GenericMask


class ExistingCapacityMask(GenericMask):
    """Capacity committed outside the current optimization decisions."""

    name = "existing_capacity"
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def get_mask(cls, model_constructor, registry):
        existing = model_constructor.optimization_model.parameters[ExistingCapacities]
        return existing.fillna(0) > 0
