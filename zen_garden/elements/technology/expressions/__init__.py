"""Technology expressions."""

from zen_garden.model.component_types.expression import GenericExpression

from .annual_fixed_opex import AnnualFixedOpex
from .annual_technology_emissions import AnnualTechnologyEmissions
from .annual_variable_opex import AnnualVariableOpex
from .annualized_capex_from_existing_capacity import AnnualizedCapexFromExistingCapacity
from .annualized_capex_from_new_investments import AnnualizedCapexFromNewInvestments
from .capacity_additions_from_new_investments import CapacityAdditionsFromNewInvestments
from .capacity_additions_from_pre_horizon_investments import (
    CapacityAdditionsFromPreHorizonInvestments,
)
from .capacity_growth_from_existing_capacity import CapacityGrowthFromExistingCapacity
from .capacity_growth_from_existing_capacity_with_spillovers import (
    CapacityGrowthFromExistingCapacityWithSpillovers,
)
from .capacity_growth_from_local_learning import CapacityGrowthFromLocalLearning
from .capacity_growth_from_spillovers import CapacityGrowthFromSpillovers
from .capacity_growth_from_unbounded_market import CapacityGrowthFromUnboundedMarket
from .existing_capacity_is_below_limit import ExistingCapacityIsBelowLimit
from .investments_beyond_horizon import InvestmentsBeyondHorizon
from .on_off_capacity_limit import OnOffCapacityLimit
from .surviving_capacity_additions import SurvivingCapacityAdditions
from .technology_activity import TechnologyActivity

TECHNOLOGY_EXPRESSIONS: list[type[GenericExpression]] = [
    SurvivingCapacityAdditions,
    CapacityAdditionsFromNewInvestments,
    CapacityAdditionsFromPreHorizonInvestments,
    CapacityGrowthFromLocalLearning,
    CapacityGrowthFromSpillovers,
    CapacityGrowthFromUnboundedMarket,
    CapacityGrowthFromExistingCapacity,
    CapacityGrowthFromExistingCapacityWithSpillovers,
    InvestmentsBeyondHorizon,
    ExistingCapacityIsBelowLimit,
    TechnologyActivity,
    AnnualTechnologyEmissions,
    AnnualVariableOpex,
    AnnualFixedOpex,
    AnnualizedCapexFromNewInvestments,
    AnnualizedCapexFromExistingCapacity,
    OnOffCapacityLimit,
]

__all__ = [
    "AnnualFixedOpex",
    "AnnualTechnologyEmissions",
    "AnnualVariableOpex",
    "AnnualizedCapexFromExistingCapacity",
    "AnnualizedCapexFromNewInvestments",
    "CapacityAdditionsFromNewInvestments",
    "CapacityAdditionsFromPreHorizonInvestments",
    "CapacityGrowthFromExistingCapacity",
    "CapacityGrowthFromExistingCapacityWithSpillovers",
    "CapacityGrowthFromLocalLearning",
    "CapacityGrowthFromSpillovers",
    "CapacityGrowthFromUnboundedMarket",
    "ExistingCapacityIsBelowLimit",
    "InvestmentsBeyondHorizon",
    "OnOffCapacityLimit",
    "SurvivingCapacityAdditions",
    "TECHNOLOGY_EXPRESSIONS",
    "TechnologyActivity",
]
