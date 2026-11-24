"""Energy system variables."""

from zen_garden.model.component_types.variable import GenericVariable

from .carbon_emissions_annual_overshoot import CarbonEmissionsAnnualOvershoot
from .carbon_emissions_budget_overshoot import CarbonEmissionsBudgetOvershoot

ENERGY_SYSTEM_VARIABLES: list[type[GenericVariable]] = [
    CarbonEmissionsBudgetOvershoot,
    CarbonEmissionsAnnualOvershoot,
]

__all__ = [
    "CarbonEmissionsBudgetOvershoot",
    "CarbonEmissionsAnnualOvershoot",
    "ENERGY_SYSTEM_VARIABLES",
]
