"""Storage technology variables."""

from zen_garden.model.component_types.variable import GenericVariable

from .capacity_addition_energy import CapacityAdditionEnergy
from .capacity_energy import CapacityEnergy
from .capacity_investment_energy import CapacityInvestmentEnergy
from .capacity_previous_energy import CapacityPreviousEnergy
from .charge_storage_binary import ChargeStorageBinary
from .cost_capex_overnight_energy import CostCapexOvernightEnergy
from .cost_capex_yearly_energy import CostCapexYearlyEnergy
from .flow_storage_charge import FlowStorageCharge
from .flow_storage_discharge import FlowStorageDischarge
from .flow_storage_spillage import FlowStorageSpillage
from .storage_level import StorageLevel
from .technology_installation_energy import TechnologyInstallationEnergy

STORAGE_TECHNOLOGY_VARIABLES: list[type[GenericVariable]] = [
    CapacityEnergy,
    CapacityPreviousEnergy,
    CapacityAdditionEnergy,
    CapacityInvestmentEnergy,
    CostCapexOvernightEnergy,
    CostCapexYearlyEnergy,
    TechnologyInstallationEnergy,
    FlowStorageCharge,
    FlowStorageDischarge,
    StorageLevel,
    FlowStorageSpillage,
    ChargeStorageBinary,
]

__all__ = [
    "CapacityEnergy",
    "CapacityPreviousEnergy",
    "CapacityAdditionEnergy",
    "CapacityInvestmentEnergy",
    "CostCapexOvernightEnergy",
    "CostCapexYearlyEnergy",
    "TechnologyInstallationEnergy",
    "FlowStorageCharge",
    "FlowStorageDischarge",
    "StorageLevel",
    "FlowStorageSpillage",
    "ChargeStorageBinary",
    "STORAGE_TECHNOLOGY_VARIABLES",
]
