"""Installed-capacity upper-limit constraint."""

import numpy as np

from zen_garden.elements.technology.expressions.existing_capacity_is_below_limit import (  # noqa: E501
    ExistingCapacityIsBelowLimit,
)
from zen_garden.elements.technology.masks import CapacityAvailableMask
from zen_garden.elements.technology.parameters.capacity_limit import CapacityLimit
from zen_garden.elements.technology.variables.capacity import Capacity
from zen_garden.model.component_types.constraint import GenericConstraint


class TechnologyCapacityLimitNotReachedConstraint(GenericConstraint):
    """Limit capacity where existing assets remain below a finite limit."""

    name = "constraint_technology_capacity_limit_not_reached"
    indices = ("set_technologies", "set_location", "set_years")
    masks = (CapacityAvailableMask,)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct limit capacity where existing assets remain below a finite limit.

        Summary:
            Limit capacity where existing assets remain below a finite limit.

        Formulation:

        .. math::

            K\\leq\\bar k\\quad\\text{if }k^{existing}<\\bar k<\\infty

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
        limit = optimization_model.parameters[CapacityLimit]
        selected = (limit != np.inf) & optimization_model.expressions[
            ExistingCapacityIsBelowLimit
        ]
        return optimization_model.variables[Capacity].where(selected) <= limit.where(
            selected, 0.0
        )
