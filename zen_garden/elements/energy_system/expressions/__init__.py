"""Energy system expressions."""

from zen_garden.model.component_types.expression import GenericExpression

from .annual_carbon_cost import AnnualCarbonCost
from .cumulative_emissions_change import CumulativeEmissionsChange
from .discounted_annual_system_cost import DiscountedAnnualSystemCost
from .emissions_counted_toward_budget import EmissionsCountedTowardBudget
from .total_carbon_emissions import TotalCarbonEmissions
from .total_cost import TotalCost

ENERGY_SYSTEM_EXPRESSIONS: list[type[GenericExpression]] = [
    DiscountedAnnualSystemCost,
    AnnualCarbonCost,
    CumulativeEmissionsChange,
    EmissionsCountedTowardBudget,
    TotalCost,
    TotalCarbonEmissions,
]

__all__ = [
    "AnnualCarbonCost",
    "CumulativeEmissionsChange",
    "DiscountedAnnualSystemCost",
    "EmissionsCountedTowardBudget",
    "ENERGY_SYSTEM_EXPRESSIONS",
    "TotalCarbonEmissions",
    "TotalCost",
]
