"""Energy system expressions."""

from zen_garden.model.component_types.expression import GenericExpression

from .annual_carbon_cost import AnnualCarbonCost
from .carbon_emissions_annual import CarbonEmissionsAnnual
from .carbon_emissions_cumulative import CarbonEmissionsCumulative
from .cost_carbon_emissions_total import CostCarbonEmissionsTotal
from .cost_total import CostTotal
from .discounted_annual_system_cost import DiscountedAnnualSystemCost
from .emissions_counted_toward_budget import EmissionsCountedTowardBudget
from .net_present_cost import NetPresentCost
from .total_carbon_emissions import TotalCarbonEmissions
from .total_cost import TotalCost

ENERGY_SYSTEM_EXPRESSIONS: list[type[GenericExpression]] = [
    CarbonEmissionsAnnual,
    CarbonEmissionsCumulative,
    DiscountedAnnualSystemCost,
    AnnualCarbonCost,
    CostCarbonEmissionsTotal,
    CostTotal,
    EmissionsCountedTowardBudget,
    NetPresentCost,
    TotalCost,
    TotalCarbonEmissions,
]

__all__ = [
    "AnnualCarbonCost",
    "CarbonEmissionsAnnual",
    "CarbonEmissionsCumulative",
    "CostCarbonEmissionsTotal",
    "CostTotal",
    "DiscountedAnnualSystemCost",
    "EmissionsCountedTowardBudget",
    "ENERGY_SYSTEM_EXPRESSIONS",
    "NetPresentCost",
    "TotalCarbonEmissions",
    "TotalCost",
]
