"""Storage-technology parameters."""

from zen_garden.model.component_types.parameter import GenericParameter

from .capacity_addition_max_energy import CapacityAdditionMaxEnergy
from .capacity_addition_min_energy import CapacityAdditionMinEnergy
from .capacity_addition_unbounded_energy import CapacityAdditionUnboundedEnergy
from .capacity_existing_energy import CapacityExistingEnergy
from .capacity_investment_existing_energy import CapacityInvestmentExistingEnergy
from .capacity_limit_energy import CapacityLimitEnergy
from .capacity_lower_limit_energy import CapacityLowerLimitEnergy
from .capex_capacity_existing_energy import CapexCapacityExistingEnergy
from .capex_specific_storage import CapexSpecificStorage
from .capex_specific_storage_energy import CapexSpecificStorageEnergy
from .efficiency_charge import EfficiencyCharge
from .efficiency_discharge import EfficiencyDischarge
from .energy_to_power_ratio_max import EnergyToPowerRatioMax
from .energy_to_power_ratio_min import EnergyToPowerRatioMin
from .existing_capacities_energy import ExistingCapacitiesEnergy
from .existing_capex_energy import ExistingCapexEnergy
from .flow_storage_inflow import FlowStorageInflow
from .opex_specific_fixed_energy import OpexSpecificFixedEnergy
from .self_discharge import SelfDischarge

STORAGE_TECHNOLOGY_PARAMETERS: list[type[GenericParameter]] = [
    CapacityExistingEnergy,
    CapacityInvestmentExistingEnergy,
    CapacityAdditionMinEnergy,
    CapacityAdditionMaxEnergy,
    CapacityAdditionUnboundedEnergy,
    CapacityLimitEnergy,
    CapacityLowerLimitEnergy,
    OpexSpecificFixedEnergy,
    CapexCapacityExistingEnergy,
    ExistingCapacitiesEnergy,
    ExistingCapexEnergy,
    EnergyToPowerRatioMin,
    EnergyToPowerRatioMax,
    EfficiencyCharge,
    EfficiencyDischarge,
    FlowStorageInflow,
    SelfDischarge,
    CapexSpecificStorage,
    CapexSpecificStorageEnergy,
]

__all__ = [parameter.__name__ for parameter in STORAGE_TECHNOLOGY_PARAMETERS]
__all__.append("STORAGE_TECHNOLOGY_PARAMETERS")
