"""Energy system constraints."""

from zen_garden.model.component_types.constraint import GenericConstraint

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

ENERGY_SYSTEM_CONSTRAINTS: list[type[GenericConstraint]] = [
    CarbonEmissionsAnnualLimitConstraint,
    CarbonEmissionsBudgetConstraint,
    CarbonEmissionsBudgetOvershootConstraint,
    CarbonEmissionsAnnualOvershootConstraint,
]

__all__ = [
    "CarbonEmissionsAnnualLimitConstraint",
    "CarbonEmissionsAnnualOvershootConstraint",
    "CarbonEmissionsBudgetConstraint",
    "CarbonEmissionsBudgetOvershootConstraint",
]
