from zen_garden.constraints.generic_constraint import GenericConstraint

from .carbon_emissions_annual_constraint import CarbonEmissionsAnnualConstraint
from .carbon_emissions_annual_limit_constraint import (
    CarbonEmissionsAnnualLimitConstraint,
)
from .carbon_emissions_annual_overshoot_constraint import (
    CarbonEmissionsAnnualOvershootConstraint,
)
from .carbon_emissions_budget_constraint import CarbonEmissionsBudgetConstraint
from .carbon_emissions_budget_overshoot_constraint import (
    CarbonEmissionsBudgetOvershootConstraint,
)
from .carbon_emissions_cumulative_constraint import CarbonEmissionsCumulativeConstraint
from .cost_carbon_emissions_total_constraint import CostCarbonEmissionsTotalConstraint
from .cost_total_constraint import CostTotalConstraint
from .net_present_cost_constraint import NetPresentCostConstraint

ENERGY_SYSTEM_CONSTRAINTS: list[type[GenericConstraint]] = [
    CarbonEmissionsCumulativeConstraint,
    CarbonEmissionsAnnualLimitConstraint,
    CarbonEmissionsBudgetConstraint,
    NetPresentCostConstraint,
    CarbonEmissionsAnnualConstraint,
    CostCarbonEmissionsTotalConstraint,
    CostTotalConstraint,
    CarbonEmissionsBudgetOvershootConstraint,
    CarbonEmissionsAnnualOvershootConstraint,
]

__all__ = [
    "CarbonEmissionsAnnualConstraint",
    "CarbonEmissionsAnnualLimitConstraint",
    "CarbonEmissionsAnnualOvershootConstraint",
    "CarbonEmissionsBudgetConstraint",
    "CarbonEmissionsBudgetOvershootConstraint",
    "CarbonEmissionsCumulativeConstraint",
    "CostCarbonEmissionsTotalConstraint",
    "CostTotalConstraint",
    "NetPresentCostConstraint",
]
