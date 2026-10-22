"""Mask for storage operation."""

from zen_garden.elements.storage_technology.masks.storage_inflow_mask import (
    StorageInflowMask,
)
from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.model.component_types.mask import GenericMask


class StorageOperationMask(GenericMask):
    """Storage operation required by capacity or exogenous inflow."""

    name = "storage_operation"
    dependencies = [PowerOperationMask, StorageInflowMask]

    @classmethod
    def build(cls, model_constructor, registry):
        return registry[PowerOperationMask] | registry[StorageInflowMask]
