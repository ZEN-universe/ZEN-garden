"""Mask for active annual CAPEX."""

from zen_garden.elements.technology.masks.existing_capex_mask import ExistingCapexMask
from zen_garden.elements.technology.masks.modeled_capex_mask import ModeledCapexMask
from zen_garden.model.component_types.mask import GenericMask


class CapexActiveMask(GenericMask):
    """Existing or modeled annual CAPEX active in a model year."""

    name = "capex_active"
    dependencies = [ExistingCapexMask, ModeledCapexMask]

    @classmethod
    def build(cls, model_constructor, registry):
        return registry[ExistingCapexMask] | registry[ModeledCapexMask]
