"""Structural masks for storage technology model components."""

from zen_garden.model.component_types.mask import GenericMask

from .storage_inflow_mask import StorageInflowMask
from .storage_operation_mask import StorageOperationMask
from .storage_state_mask import StorageStateMask

STORAGE_TECHNOLOGY_MASKS: list[type[GenericMask]] = [
    StorageInflowMask,
    StorageOperationMask,
    StorageStateMask,
]

__all__ = [
    "StorageInflowMask",
    "StorageOperationMask",
    "StorageStateMask",
    "STORAGE_TECHNOLOGY_MASKS",
]
