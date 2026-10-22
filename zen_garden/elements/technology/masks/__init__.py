"""Structural masks shared by technology model components."""

from zen_garden.model.component_types.mask import GenericMask

from .capacity_addition_any_type_mask import CapacityAdditionAnyTypeMask
from .capacity_addition_eligible_mask import CapacityAdditionEligibleMask
from .capacity_any_type_mask import CapacityAnyTypeMask
from .capacity_available_mask import CapacityAvailableMask
from .capex_active_mask import CapexActiveMask
from .existing_capacity_mask import ExistingCapacityMask
from .existing_capex_mask import ExistingCapexMask
from .installation_relevant_mask import InstallationRelevantMask
from .investment_eligible_mask import InvestmentEligibleMask
from .modeled_capex_mask import ModeledCapexMask
from .on_off_technology_mask import OnOffTechnologyMask
from .potential_new_capacity_mask import PotentialNewCapacityMask
from .power_capacity_addition_mask import PowerCapacityAdditionMask
from .power_capacity_year_mask import PowerCapacityYearMask
from .power_operation_mask import PowerOperationMask

TECHNOLOGY_MASKS: list[type[GenericMask]] = [
    ExistingCapacityMask,
    CapacityAdditionEligibleMask,
    InvestmentEligibleMask,
    PowerCapacityAdditionMask,
    CapacityAdditionAnyTypeMask,
    PotentialNewCapacityMask,
    CapacityAvailableMask,
    CapacityAnyTypeMask,
    PowerCapacityYearMask,
    ExistingCapexMask,
    ModeledCapexMask,
    CapexActiveMask,
    PowerOperationMask,
    OnOffTechnologyMask,
    InstallationRelevantMask,
]

__all__ = [
    "CapacityAdditionAnyTypeMask",
    "CapacityAdditionEligibleMask",
    "CapacityAnyTypeMask",
    "CapacityAvailableMask",
    "CapexActiveMask",
    "ExistingCapacityMask",
    "ExistingCapexMask",
    "InstallationRelevantMask",
    "InvestmentEligibleMask",
    "ModeledCapexMask",
    "OnOffTechnologyMask",
    "PotentialNewCapacityMask",
    "PowerCapacityAdditionMask",
    "PowerCapacityYearMask",
    "PowerOperationMask",
    "TECHNOLOGY_MASKS",
]
