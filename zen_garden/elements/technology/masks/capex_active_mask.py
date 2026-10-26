"""Mask for active annual CAPEX."""

from zen_garden.elements.technology.masks.existing_capex_mask import ExistingCapexMask
from zen_garden.elements.technology.masks.modeled_capex_mask import ModeledCapexMask
from zen_garden.model.component_types.mask import GenericMask


class CapexActiveMask(GenericMask):
    """Existing or modeled annual CAPEX active in a model year."""

    name = "capex_active"
    indices = ("set_technologies", "set_location", "set_years")
    dependencies = (ExistingCapexMask, ModeledCapexMask)

    @classmethod
    def get_mask(cls, model_constructor, registry):
        return registry[ExistingCapexMask] | registry[ModeledCapexMask]
