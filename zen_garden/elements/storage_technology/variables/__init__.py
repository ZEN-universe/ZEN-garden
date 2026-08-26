"""Storage technology variables."""

from zen_garden.topology.generic_variable import GenericVariable

from .charge_storage_binary import ChargeStorageBinary
from .flow_storage_charge import FlowStorageCharge
from .flow_storage_discharge import FlowStorageDischarge
from .flow_storage_spillage import FlowStorageSpillage
from .storage_level import StorageLevel

STORAGE_TECHNOLOGY_VARIABLES: list[type[GenericVariable]] = [
    FlowStorageCharge,
    FlowStorageDischarge,
    StorageLevel,
    FlowStorageSpillage,
    ChargeStorageBinary,
]

__all__ = [
    "FlowStorageCharge",
    "FlowStorageDischarge",
    "StorageLevel",
    "FlowStorageSpillage",
    "ChargeStorageBinary",
    "STORAGE_TECHNOLOGY_VARIABLES",
]

