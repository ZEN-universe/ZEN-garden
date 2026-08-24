"""Storage technology constraints."""

from zen_garden.constraints.generic_constraint import GenericConstraint

from .capacity_energy_to_power_ratio_constraint import (
    CapacityEnergyToPowerRatioConstraint,
)
from .capacity_factor_storage_constraint import CapacityFactorStorageConstraint
from .charge_discharge_binary_constraint import ChargeDischargeBinaryConstraint
from .couple_storage_level_constraint import CoupleStorageLevelConstraint
from .flow_storage_spillage_constraint import FlowStorageSpillageConstraint
from .opex_emissions_technology_storage_constraint import (
    OpexEmissionsTechnologyStorageConstraint,
)
from .storage_level_max_constraint import StorageLevelMaxConstraint
from .storage_technology_capex_constraint import StorageTechnologyCapexConstraint

STORAGE_TECHNOLOGY_CONSTRAINTS: list[type[GenericConstraint]] = [
    CapacityFactorStorageConstraint,
    OpexEmissionsTechnologyStorageConstraint,
    StorageLevelMaxConstraint,
    CoupleStorageLevelConstraint,
    FlowStorageSpillageConstraint,
    CapacityEnergyToPowerRatioConstraint,
    StorageTechnologyCapexConstraint,
    ChargeDischargeBinaryConstraint,
]

__all__ = [
    "CapacityEnergyToPowerRatioConstraint",
    "CapacityFactorStorageConstraint",
    "ChargeDischargeBinaryConstraint",
    "CoupleStorageLevelConstraint",
    "FlowStorageSpillageConstraint",
    "OpexEmissionsTechnologyStorageConstraint",
    "StorageLevelMaxConstraint",
    "StorageTechnologyCapexConstraint",
]
