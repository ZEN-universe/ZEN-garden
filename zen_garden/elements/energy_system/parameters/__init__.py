"""energy system parameters."""

from zen_garden.model.component_types.parameter import GenericParameter

from .carbon_emissions_annual_limit import CarbonEmissionsAnnualLimit
from .carbon_emissions_budget import CarbonEmissionsBudget
from .carbon_emissions_cumulative_existing import CarbonEmissionsCumulativeExisting
from .discount_rate import DiscountRate
from .duration_representative_hours_storage import (
    DurationRepresentativeHoursStorage,
)
from .interval_between_years import IntervalBetweenYears
from .knowledge_depreciation_rate import KnowledgeDepreciationRate
from .knowledge_spillover_rate import KnowledgeSpilloverRate
from .market_share_unbounded import MarketShareUnbounded
from .modeled_year_hours import ModeledYearHours
from .price_carbon_emissions import PriceCarbonEmissions
from .price_carbon_emissions_annual_overshoot import PriceCarbonEmissionsAnnualOvershoot
from .price_carbon_emissions_budget_overshoot import PriceCarbonEmissionsBudgetOvershoot
from .weight_representative_hours import WeightRepresentativeHours

ENERGY_SYSTEM_PARAMETERS: list[type[GenericParameter]] = [
    IntervalBetweenYears,
    ModeledYearHours,
    WeightRepresentativeHours,
    DurationRepresentativeHoursStorage,
    DiscountRate,
    CarbonEmissionsAnnualLimit,
    CarbonEmissionsBudget,
    CarbonEmissionsCumulativeExisting,
    PriceCarbonEmissions,
    PriceCarbonEmissionsBudgetOvershoot,
    PriceCarbonEmissionsAnnualOvershoot,
    MarketShareUnbounded,
    KnowledgeDepreciationRate,
    KnowledgeSpilloverRate,
]

__all__ = [
    "IntervalBetweenYears",
    "ModeledYearHours",
    "WeightRepresentativeHours",
    "DurationRepresentativeHoursStorage",
    "DiscountRate",
    "CarbonEmissionsAnnualLimit",
    "CarbonEmissionsBudget",
    "CarbonEmissionsCumulativeExisting",
    "PriceCarbonEmissions",
    "PriceCarbonEmissionsBudgetOvershoot",
    "PriceCarbonEmissionsAnnualOvershoot",
    "MarketShareUnbounded",
    "KnowledgeDepreciationRate",
    "KnowledgeSpilloverRate",
    "ENERGY_SYSTEM_PARAMETERS",
]
