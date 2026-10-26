"""Capacity-addition restriction at reached capacity limits."""

import numpy as np

from zen_garden.elements.technology.expressions.existing_capacity_is_below_limit import (  # noqa: E501
    ExistingCapacityIsBelowLimit,
)
from zen_garden.elements.technology.masks import CapacityAdditionEligibleMask
from zen_garden.elements.technology.parameters.capacity_limit import CapacityLimit
from zen_garden.elements.technology.variables.capacity_addition import CapacityAddition
from zen_garden.model.component_types.constraint import GenericConstraint


class TechnologyCapacityLimitReachedConstraint(GenericConstraint):
    """Prevent additions where existing capacity already reaches the limit."""

    name = "constraint_technology_capacity_limit_reached"
    indices = ("set_technologies", "set_location", "set_years")
    masks = (CapacityAdditionEligibleMask,)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct the model component.

        Summary:
            Prevent additions where existing capacity already reaches the limit.

        Formulation:

        .. math::

            \\Delta K=0\\quad\\text{if }k^{existing}\\geq\\bar k

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        optimization_model = model_constructor.optimization_model
        additions = optimization_model.variables[CapacityAddition]
        if not model_constructor.config.system.allow_investment:
            return additions == 0
        limit = optimization_model.parameters[CapacityLimit]
        selected = (limit != np.inf) & ~optimization_model.expressions[
            ExistingCapacityIsBelowLimit
        ]
        return additions.where(selected) == 0
