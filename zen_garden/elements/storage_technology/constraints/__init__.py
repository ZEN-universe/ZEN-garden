"""Storage technology constraints."""

from zen_garden.model.component_types.constraint import GenericConstraint

from .capacity_energy_to_power_ratio_max_constraint import (
    CapacityEnergyToPowerRatioMaxConstraint,
)
from .capacity_energy_to_power_ratio_min_constraint import (
    CapacityEnergyToPowerRatioMinConstraint,
)
from .capacity_factor_storage_constraint import CapacityFactorStorageConstraint
from .charge_storage_binary_constraint import ChargeStorageBinaryConstraint
from .couple_storage_level_constraint import CoupleStorageLevelConstraint
from .discharge_storage_binary_constraint import DischargeStorageBinaryConstraint
from .flow_storage_spillage_constraint import FlowStorageSpillageConstraint
from .storage_energy_capacity_limit_not_reached_constraint import (
    StorageEnergyCapacityLimitNotReachedConstraint,
)
from .storage_energy_capacity_limit_reached_constraint import (
    StorageEnergyCapacityLimitReachedConstraint,
)
from .storage_energy_capacity_lower_limit_constraint import (
    StorageEnergyCapacityLowerLimitConstraint,
)
from .storage_energy_construction_time_constraint import (
    StorageEnergyConstructionTimeConstraint,
)
from .storage_energy_construction_time_outside_constraint import (
    StorageEnergyConstructionTimeOutsideConstraint,
)
from .storage_energy_diffusion_limit_location_constraint import (
    StorageEnergyDiffusionLimitLocationConstraint,
)
from .storage_energy_diffusion_limit_total_constraint import (
    StorageEnergyDiffusionLimitTotalConstraint,
)
from .storage_energy_lifetime_constraint import StorageEnergyLifetimeConstraint
from .storage_energy_lifetime_previous_constraint import (
    StorageEnergyLifetimePreviousConstraint,
)
from .storage_energy_max_capacity_addition_constraint import (
    StorageEnergyMaxCapacityAdditionConstraint,
)
from .storage_energy_min_capacity_addition_constraint import (
    StorageEnergyMinCapacityAdditionConstraint,
)
from .storage_level_max_constraint import StorageLevelMaxConstraint

STORAGE_TECHNOLOGY_CONSTRAINTS: list[type[GenericConstraint]] = [
    StorageEnergyCapacityLimitNotReachedConstraint,
    StorageEnergyCapacityLimitReachedConstraint,
    StorageEnergyCapacityLowerLimitConstraint,
    StorageEnergyMinCapacityAdditionConstraint,
    StorageEnergyMaxCapacityAdditionConstraint,
    StorageEnergyConstructionTimeConstraint,
    StorageEnergyConstructionTimeOutsideConstraint,
    StorageEnergyDiffusionLimitTotalConstraint,
    StorageEnergyDiffusionLimitLocationConstraint,
    StorageEnergyLifetimeConstraint,
    StorageEnergyLifetimePreviousConstraint,
    CapacityFactorStorageConstraint,
    StorageLevelMaxConstraint,
    CoupleStorageLevelConstraint,
    FlowStorageSpillageConstraint,
    CapacityEnergyToPowerRatioMinConstraint,
    CapacityEnergyToPowerRatioMaxConstraint,
    ChargeStorageBinaryConstraint,
    DischargeStorageBinaryConstraint,
]

__all__ = [
    "StorageEnergyCapacityLimitNotReachedConstraint",
    "StorageEnergyCapacityLimitReachedConstraint",
    "StorageEnergyCapacityLowerLimitConstraint",
    "StorageEnergyMinCapacityAdditionConstraint",
    "StorageEnergyMaxCapacityAdditionConstraint",
    "StorageEnergyConstructionTimeConstraint",
    "StorageEnergyConstructionTimeOutsideConstraint",
    "StorageEnergyDiffusionLimitTotalConstraint",
    "StorageEnergyDiffusionLimitLocationConstraint",
    "StorageEnergyLifetimeConstraint",
    "StorageEnergyLifetimePreviousConstraint",
    "CapacityEnergyToPowerRatioMaxConstraint",
    "CapacityEnergyToPowerRatioMinConstraint",
    "CapacityFactorStorageConstraint",
    "ChargeStorageBinaryConstraint",
    "CoupleStorageLevelConstraint",
    "FlowStorageSpillageConstraint",
    "DischargeStorageBinaryConstraint",
    "StorageLevelMaxConstraint",
]
