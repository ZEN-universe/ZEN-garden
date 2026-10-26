"""Structural masks for storage technology model components."""

from zen_garden.model.component_types.mask import GenericMask

from .energy_capacity_addition_eligible_mask import EnergyCapacityAdditionEligibleMask
from .energy_capacity_available_mask import EnergyCapacityAvailableMask
from .energy_capex_active_mask import EnergyCapexActiveMask
from .energy_existing_capacity_mask import EnergyExistingCapacityMask
from .energy_existing_capex_mask import EnergyExistingCapexMask
from .energy_installation_relevant_mask import EnergyInstallationRelevantMask
from .energy_investment_eligible_mask import EnergyInvestmentEligibleMask
from .energy_modeled_capex_mask import EnergyModeledCapexMask
from .energy_potential_new_capacity_mask import EnergyPotentialNewCapacityMask
from .storage_inflow_mask import StorageInflowMask
from .storage_operation_mask import StorageOperationMask
from .storage_state_mask import StorageStateMask

STORAGE_TECHNOLOGY_MASKS: list[type[GenericMask]] = [
    EnergyExistingCapacityMask,
    EnergyCapacityAdditionEligibleMask,
    EnergyPotentialNewCapacityMask,
    EnergyCapacityAvailableMask,
    EnergyInvestmentEligibleMask,
    EnergyInstallationRelevantMask,
    EnergyExistingCapexMask,
    EnergyModeledCapexMask,
    EnergyCapexActiveMask,
    StorageInflowMask,
    StorageOperationMask,
    StorageStateMask,
]

__all__ = [
    "EnergyExistingCapacityMask",
    "EnergyCapacityAdditionEligibleMask",
    "EnergyPotentialNewCapacityMask",
    "EnergyCapacityAvailableMask",
    "EnergyInvestmentEligibleMask",
    "EnergyInstallationRelevantMask",
    "EnergyExistingCapexMask",
    "EnergyModeledCapexMask",
    "EnergyCapexActiveMask",
    "StorageInflowMask",
    "StorageOperationMask",
    "StorageStateMask",
    "STORAGE_TECHNOLOGY_MASKS",
]
