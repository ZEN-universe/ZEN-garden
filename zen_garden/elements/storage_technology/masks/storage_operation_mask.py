"""Mask for storage operation."""

from zen_garden.elements.storage_technology.masks.storage_inflow_mask import (
    StorageInflowMask,
)
from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.model.component_types.mask import GenericMask


class StorageOperationMask(GenericMask):
    """Storage operation required by capacity or exogenous inflow."""

    name = "storage_operation"
    indices = (
        "set_technologies",
        "set_location",
        "set_years",
        "set_representative_hours",
    )
    dependencies = (PowerOperationMask, StorageInflowMask)

    @classmethod
    def get_mask(cls, model_constructor, registry):
        return registry[PowerOperationMask] | registry[StorageInflowMask]
