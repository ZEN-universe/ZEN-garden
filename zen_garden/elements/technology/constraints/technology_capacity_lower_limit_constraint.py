from zen_garden.elements.technology.masks import CapacityAvailableMask
from zen_garden.elements.technology.parameters.capacity_lower_limit import (
    CapacityLowerLimit,
)
from zen_garden.elements.technology.variables.capacity import Capacity
from zen_garden.model.component_types.constraint import GenericConstraint


class TechnologyCapacityLowerLimitConstraint(GenericConstraint):
    """Technology Capacity Lower Limit."""

    name = "constraint_technology_capacity_lower_limit"
    indices = ("set_technologies", "set_location", "set_years")
    masks = (CapacityAvailableMask,)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Constrain installed capacity by each positive lower limit.

        Formulation:

        .. math::
            K_{h,p,y} \\geq \\underline{k}_{h,p,y}

        No constraint is created where the configured lower limit is zero.
        For storage technologies, the equation is applied independently to power
        and energy capacity.

        Notation:

        :math:`K_{h,p,y}`: installed capacity of technology :math:`h` at
        location :math:`p` in year :math:`y`
        :math:`\\underline{k}_{h,p,y}`: lower capacity limit

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """

        # In TechnologyRules, we access variables and parameters directly via self
        capacity = model_constructor.optimization_model.variables[Capacity]
        capacity_lower_limit = model_constructor.optimization_model.parameters[
            CapacityLowerLimit
        ]

        # Create a mask so we only build constraints
        # where the user actually provided a number
        mask = capacity_lower_limit > 0.0

        # Apply the mask using xarray's .where() so we don't build empty/NaN constraints
        lhs = capacity.where(mask)
        rhs = capacity_lower_limit.where(mask, 0.0)

        # Total Capacity >= Lower Bound
        constraint = lhs >= rhs

        # Add the constraint to the model
        return constraint
