"""Technology constraints."""

from zen_garden.constraints.generic_constraint import GenericConstraint

from .carbon_emissions_technology_total_constraint import (
    CarbonEmissionsTechnologyTotalConstraint,
)
from .cost_capex_yearly_constraint import CostCapexYearlyConstraint
from .cost_capex_yearly_total_constraint import CostCapexYearlyTotalConstraint
from .cost_opex_yearly_constraint import CostOpexYearlyConstraint
from .cost_opex_yearly_total_constraint import CostOpexYearlyTotalConstraint
from .technology_capacity_limit_constraint import TechnologyCapacityLimitConstraint
from .technology_capacity_lower_limit_constraint import (
    TechnologyCapacityLowerLimitConstraint,
)
from .technology_construction_time_constraint import (
    TechnologyConstructionTimeConstraint,
)
from .technology_diffusion_limit_constraint import TechnologyDiffusionLimitConstraint
from .technology_lifetime_constraint import TechnologyLifetimeConstraint
from .technology_max_capacity_addition_constraint import (
    TechnologyMaxCapacityAdditionConstraint,
)
from .technology_min_capacity_addition_constraint import (
    TechnologyMinCapacityAdditionConstraint,
)
from .technology_on_off_constraint import TechnologyOnOffConstraint

TECHNOLOGY_CONSTRAINTS: list[type[GenericConstraint]] = [
    TechnologyCapacityLimitConstraint,
    TechnologyCapacityLowerLimitConstraint,
    TechnologyMinCapacityAdditionConstraint,
    TechnologyMaxCapacityAdditionConstraint,
    TechnologyConstructionTimeConstraint,
    TechnologyLifetimeConstraint,
    CostOpexYearlyTotalConstraint,
    CostCapexYearlyTotalConstraint,
    CostOpexYearlyConstraint,
    CarbonEmissionsTechnologyTotalConstraint,
]

__all__ = [
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
