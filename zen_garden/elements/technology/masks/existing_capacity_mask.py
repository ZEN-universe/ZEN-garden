"""Mask for capacity committed outside the optimization decisions."""

from zen_garden.model.component_types.mask import GenericMask


class ExistingCapacityMask(GenericMask):
    """Capacity committed outside the current optimization decisions."""

    name = "existing_capacity"

    @classmethod
    def build(cls, model_constructor, registry):
        existing = model_constructor.optimization_model.parameters.existing_capacities
        return existing.fillna(0) > 0
