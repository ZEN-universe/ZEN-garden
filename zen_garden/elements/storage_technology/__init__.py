"""Storage Technology constraints."""

from zen_garden.constraints.generic_constraint import GenericConstraint
from zen_garden.elements.storage_technology.constraints.capacity_energy_to_power_ratio_constraint import (
    CapacityEnergyToPowerRatioConstraint,
)
from zen_garden.elements.storage_technology.constraints.capacity_factor_storage_constraint import (
    CapacityFactorStorageConstraint,
)
from zen_garden.elements.storage_technology.constraints.charge_discharge_binary_constraint import (
    ChargeDischargeBinaryConstraint,
)
from zen_garden.elements.storage_technology.constraints.couple_storage_level_constraint import (
    CoupleStorageLevelConstraint,
)
from zen_garden.elements.storage_technology.constraints.flow_storage_spillage_constraint import (
    FlowStorageSpillageConstraint,
)
from zen_garden.elements.storage_technology.constraints.opex_emissions_technology_storage_constraint import (
    OpexEmissionsTechnologyStorageConstraint,
)
from zen_garden.elements.storage_technology.constraints.storage_level_max_constraint import (
    StorageLevelMaxConstraint,
)
from zen_garden.elements.storage_technology.constraints.storage_technology_capex_constraint import (
    StorageTechnologyCapexConstraint,
)
from zen_garden.elements.storage_technology.storage_technology import StorageTechnology

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
    "StorageTechnology",
    "CapacityEnergyToPowerRatioConstraint",
    "CapacityFactorStorageConstraint",
    "ChargeDischargeBinaryConstraint",
    "CoupleStorageLevelConstraint",
    "FlowStorageSpillageConstraint",
    "OpexEmissionsTechnologyStorageConstraint",
    "StorageLevelMaxConstraint",
    "StorageTechnologyCapexConstraint",
]
