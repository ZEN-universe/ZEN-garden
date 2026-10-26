"""Storage technology expressions."""

from zen_garden.model.component_types.expression import GenericExpression

from .annualized_capex_from_existing_capacity_energy import (
    AnnualizedCapexFromExistingCapacityEnergy,
)
from .annualized_capex_from_new_investments_energy import (
    AnnualizedCapexFromNewInvestmentsEnergy,
)
from .capacity_additions_from_new_investments_energy import (
    CapacityAdditionsFromNewInvestmentsEnergy,
)
from .capacity_additions_from_pre_horizon_investments_energy import (
    CapacityAdditionsFromPreHorizonInvestmentsEnergy,
)
from .capacity_growth_from_existing_capacity_energy import (
    CapacityGrowthFromExistingCapacityEnergy,
)
from .capacity_growth_from_existing_capacity_with_spillovers_energy import (
    CapacityGrowthFromExistingCapacityWithSpilloversEnergy,
)
from .capacity_growth_from_local_learning_energy import (
    CapacityGrowthFromLocalLearningEnergy,
)
from .capacity_growth_from_spillovers_energy import CapacityGrowthFromSpilloversEnergy
from .capacity_growth_from_unbounded_market_energy import (
    CapacityGrowthFromUnboundedMarketEnergy,
)
from .charging_mode_capacity_limit import ChargingModeCapacityLimit
from .existing_capacity_is_below_limit_energy import (
    ExistingCapacityIsBelowLimitEnergy,
)
from .investments_beyond_horizon_energy import InvestmentsBeyondHorizonEnergy
from .storage_energy_capacity_addition import StorageEnergyCapacityAddition
from .storage_level_change_after_self_discharge import (
    StorageLevelChangeAfterSelfDischarge,
)
from .storage_net_energy_inflow import StorageNetEnergyInflow
from .storage_overnight_capex import StorageOvernightCapex
from .storage_overnight_capex_energy import StorageOvernightCapexEnergy
from .storage_power_capacity_addition import StoragePowerCapacityAddition
from .storage_power_capacity_limit import StoragePowerCapacityLimit
from .storage_throughput import StorageThroughput
from .surviving_capacity_additions_energy import SurvivingCapacityAdditionsEnergy

STORAGE_TECHNOLOGY_EXPRESSIONS: list[type[GenericExpression]] = [
    SurvivingCapacityAdditionsEnergy,
    CapacityAdditionsFromNewInvestmentsEnergy,
    CapacityAdditionsFromPreHorizonInvestmentsEnergy,
    CapacityGrowthFromLocalLearningEnergy,
    CapacityGrowthFromSpilloversEnergy,
    CapacityGrowthFromUnboundedMarketEnergy,
    CapacityGrowthFromExistingCapacityEnergy,
    CapacityGrowthFromExistingCapacityWithSpilloversEnergy,
    InvestmentsBeyondHorizonEnergy,
    ExistingCapacityIsBelowLimitEnergy,
    AnnualizedCapexFromNewInvestmentsEnergy,
    AnnualizedCapexFromExistingCapacityEnergy,
    StorageThroughput,
    StoragePowerCapacityAddition,
    StorageEnergyCapacityAddition,
    StoragePowerCapacityLimit,
    ChargingModeCapacityLimit,
    StorageLevelChangeAfterSelfDischarge,
    StorageNetEnergyInflow,
    StorageOvernightCapex,
    StorageOvernightCapexEnergy,
]

__all__ = [
    "SurvivingCapacityAdditionsEnergy",
    "CapacityAdditionsFromNewInvestmentsEnergy",
    "CapacityAdditionsFromPreHorizonInvestmentsEnergy",
    "CapacityGrowthFromLocalLearningEnergy",
    "CapacityGrowthFromSpilloversEnergy",
    "CapacityGrowthFromUnboundedMarketEnergy",
    "CapacityGrowthFromExistingCapacityEnergy",
    "CapacityGrowthFromExistingCapacityWithSpilloversEnergy",
    "InvestmentsBeyondHorizonEnergy",
    "ExistingCapacityIsBelowLimitEnergy",
    "AnnualizedCapexFromNewInvestmentsEnergy",
    "AnnualizedCapexFromExistingCapacityEnergy",
    "ChargingModeCapacityLimit",
    "STORAGE_TECHNOLOGY_EXPRESSIONS",
    "StorageEnergyCapacityAddition",
    "StorageLevelChangeAfterSelfDischarge",
    "StorageNetEnergyInflow",
    "StorageOvernightCapex",
    "StorageOvernightCapexEnergy",
    "StoragePowerCapacityAddition",
    "StoragePowerCapacityLimit",
    "StorageThroughput",
]
