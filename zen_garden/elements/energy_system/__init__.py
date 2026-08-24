"""Energy System constraints."""

from zen_garden.constraints.generic_constraint import GenericConstraint
from zen_garden.elements.energy_system.constraints.carbon_emissions_annual_constraint import (
    CarbonEmissionsAnnualConstraint,
)
from zen_garden.elements.energy_system.constraints.carbon_emissions_annual_limit_constraint import (
    CarbonEmissionsAnnualLimitConstraint,
)
from zen_garden.elements.energy_system.constraints.carbon_emissions_annual_overshoot_constraint import (
    CarbonEmissionsAnnualOvershootConstraint,
)
from zen_garden.elements.energy_system.constraints.carbon_emissions_budget_constraint import (
    CarbonEmissionsBudgetConstraint,
)
from zen_garden.elements.energy_system.constraints.carbon_emissions_budget_overshoot_constraint import (
    CarbonEmissionsBudgetOvershootConstraint,
)
from zen_garden.elements.energy_system.constraints.carbon_emissions_cumulative_constraint import (
    CarbonEmissionsCumulativeConstraint,
)
from zen_garden.elements.energy_system.constraints.cost_carbon_emissions_total_constraint import (
    CostCarbonEmissionsTotalConstraint,
)
from zen_garden.elements.energy_system.constraints.cost_total_constraint import (
    CostTotalConstraint,
)
from zen_garden.elements.energy_system.constraints.net_present_cost_constraint import (
    NetPresentCostConstraint,
)
from zen_garden.elements.energy_system.energy_system import EnergySystem

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
    "EnergySystem",
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
