"""Technology constraints."""

from zen_garden.model.component_types.constraint import GenericConstraint

from .carbon_emissions_technology_total_constraint import (
    CarbonEmissionsTechnologyTotalConstraint,
)
from .cost_capex_yearly_constraint import CostCapexYearlyConstraint
from .cost_capex_yearly_total_constraint import CostCapexYearlyTotalConstraint
from .cost_opex_yearly_constraint import CostOpexYearlyConstraint
from .cost_opex_yearly_total_constraint import CostOpexYearlyTotalConstraint
from .technology_capacity_limit_not_reached_constraint import (
    TechnologyCapacityLimitNotReachedConstraint,
)
from .technology_capacity_limit_reached_constraint import (
    TechnologyCapacityLimitReachedConstraint,
)
from .technology_capacity_lower_limit_constraint import (
    TechnologyCapacityLowerLimitConstraint,
)
from .technology_construction_time_constraint import (
    TechnologyConstructionTimeConstraint,
)
from .technology_construction_time_outside_constraint import (
    TechnologyConstructionTimeOutsideConstraint,
)
from .technology_diffusion_limit_location_constraint import (
    TechnologyDiffusionLimitLocationConstraint,
)
from .technology_diffusion_limit_total_constraint import (
    TechnologyDiffusionLimitTotalConstraint,
)
from .technology_lifetime_constraint import TechnologyLifetimeConstraint
from .technology_lifetime_previous_constraint import (
    TechnologyLifetimePreviousConstraint,
)
from .technology_max_capacity_addition_constraint import (
    TechnologyMaxCapacityAdditionConstraint,
)
from .technology_min_capacity_addition_constraint import (
    TechnologyMinCapacityAdditionConstraint,
)
from .technology_on_off_capacity_helper_constraint import (
    TechnologyOnOffCapacityHelperConstraint,
)
from .technology_on_off_capacity_helper_lower_bound_constraint import (
    TechnologyOnOffCapacityHelperLowerBoundConstraint,
)
from .technology_on_off_capacity_helper_upper_bound_constraint import (
    TechnologyOnOffCapacityHelperUpperBoundConstraint,
)
from .technology_on_off_operation_lower_bound_constraint import (
    TechnologyOnOffOperationLowerBoundConstraint,
)
from .technology_on_off_operation_upper_bound_constraint import (
    TechnologyOnOffOperationUpperBoundConstraint,
)

TECHNOLOGY_CONSTRAINTS: list[type[GenericConstraint]] = [
    TechnologyCapacityLimitNotReachedConstraint,
    TechnologyCapacityLimitReachedConstraint,
    TechnologyCapacityLowerLimitConstraint,
    TechnologyMinCapacityAdditionConstraint,
    TechnologyMaxCapacityAdditionConstraint,
    TechnologyConstructionTimeConstraint,
    TechnologyConstructionTimeOutsideConstraint,
    TechnologyLifetimeConstraint,
    TechnologyLifetimePreviousConstraint,
    TechnologyDiffusionLimitTotalConstraint,
    TechnologyDiffusionLimitLocationConstraint,
    CostCapexYearlyConstraint,
    CostCapexYearlyTotalConstraint,
    CostOpexYearlyConstraint,
    CostOpexYearlyTotalConstraint,
    CarbonEmissionsTechnologyTotalConstraint,
    TechnologyOnOffOperationLowerBoundConstraint,
    TechnologyOnOffOperationUpperBoundConstraint,
    TechnologyOnOffCapacityHelperConstraint,
    TechnologyOnOffCapacityHelperLowerBoundConstraint,
    TechnologyOnOffCapacityHelperUpperBoundConstraint,
]

__all__ = [constraint.__name__ for constraint in TECHNOLOGY_CONSTRAINTS]
