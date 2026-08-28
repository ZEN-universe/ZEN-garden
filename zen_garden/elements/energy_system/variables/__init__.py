"""Energy system variables."""

from zen_garden.model.component_types.variable import GenericVariable

from .carbon_emissions_annual import CarbonEmissionsAnnual
from .carbon_emissions_annual_overshoot import CarbonEmissionsAnnualOvershoot
from .carbon_emissions_budget_overshoot import CarbonEmissionsBudgetOvershoot
from .carbon_emissions_cumulative import CarbonEmissionsCumulative
from .cost_carbon_emissions_total import CostCarbonEmissionsTotal
from .cost_total import CostTotal
from .net_present_cost import NetPresentCost

ENERGY_SYSTEM_VARIABLES: list[type[GenericVariable]] = [
    CarbonEmissionsAnnual,
    CarbonEmissionsCumulative,
    CarbonEmissionsBudgetOvershoot,
    CarbonEmissionsAnnualOvershoot,
    CostCarbonEmissionsTotal,
    CostTotal,
    NetPresentCost,
]

__all__ = [
    "CarbonEmissionsAnnual",
    "CarbonEmissionsCumulative",
    "CarbonEmissionsBudgetOvershoot",
    "CarbonEmissionsAnnualOvershoot",
    "CostCarbonEmissionsTotal",
    "CostTotal",
    "NetPresentCost",
    "ENERGY_SYSTEM_VARIABLES",
]
