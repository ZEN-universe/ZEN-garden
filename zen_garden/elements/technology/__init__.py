"""Technology constraints."""

from zen_garden.constraints.generic_constraint import GenericConstraint
from zen_garden.elements.technology.technology import Technology

from zen_garden.elements.technology.constraints.carbon_emissions_technology_total_constraint import (
    CarbonEmissionsTechnologyTotalConstraint,
)
from zen_garden.elements.technology.constraints.cost_capex_yearly_constraint import CostCapexYearlyConstraint
from zen_garden.elements.technology.constraints.cost_capex_yearly_total_constraint import CostCapexYearlyTotalConstraint
from zen_garden.elements.technology.constraints.cost_opex_yearly_constraint import CostOpexYearlyConstraint
from zen_garden.elements.technology.constraints.cost_opex_yearly_total_constraint import CostOpexYearlyTotalConstraint
from zen_garden.elements.technology.constraints.technology_capacity_limit_constraint import TechnologyCapacityLimitConstraint
from zen_garden.elements.technology.constraints.technology_capacity_lower_limit_constraint import (
    TechnologyCapacityLowerLimitConstraint,
)
from zen_garden.elements.technology.constraints.technology_construction_time_constraint import (
    TechnologyConstructionTimeConstraint,
)
from zen_garden.elements.technology.constraints.technology_diffusion_limit_constraint import TechnologyDiffusionLimitConstraint
from zen_garden.elements.technology.constraints.technology_lifetime_constraint import TechnologyLifetimeConstraint
from zen_garden.elements.technology.constraints.technology_max_capacity_addition_constraint import (
    TechnologyMaxCapacityAdditionConstraint,
)
from zen_garden.elements.technology.constraints.technology_min_capacity_addition_constraint import (
    TechnologyMinCapacityAdditionConstraint,
)
from zen_garden.elements.technology.constraints.technology_on_off_constraint import TechnologyOnOffConstraint

TECHNOLOGY_CONSTRAINTS: list[type[GenericConstraint]] = [
    TechnologyCapacityLimitConstraint,
    TechnologyCapacityLowerLimitConstraint,
    TechnologyMinCapacityAdditionConstraint,
    TechnologyMaxCapacityAdditionConstraint,
    TechnologyConstructionTimeConstraint,
    TechnologyLifetimeConstraint,
    TechnologyDiffusionLimitConstraint,
    CostCapexYearlyConstraint,
    CostCapexYearlyTotalConstraint,
    CostOpexYearlyConstraint,
    CostOpexYearlyTotalConstraint,
    CarbonEmissionsTechnologyTotalConstraint,
]

__all__ = [
    "Technology",
    "CarbonEmissionsTechnologyTotalConstraint",
    "CostCapexYearlyConstraint",
    "CostCapexYearlyTotalConstraint",
    "CostOpexYearlyConstraint",
    "CostOpexYearlyTotalConstraint",
    "TechnologyCapacityLimitConstraint",
    "TechnologyCapacityLowerLimitConstraint",
    "TechnologyConstructionTimeConstraint",
    "TechnologyDiffusionLimitConstraint",
    "TechnologyLifetimeConstraint",
    "TechnologyMaxCapacityAdditionConstraint",
    "TechnologyMinCapacityAdditionConstraint",
    "TechnologyOnOffConstraint",
]
