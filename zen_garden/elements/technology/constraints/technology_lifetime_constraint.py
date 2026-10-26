"""End-of-year available-capacity constraint."""

import linopy as lp
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.elements.technology.expressions.surviving_capacity_additions import (
    SurvivingCapacityAdditions,
)
from zen_garden.elements.technology.masks import CapacityAvailableMask
from zen_garden.elements.technology.parameters.existing_capacities import (
    ExistingCapacities,
)
from zen_garden.elements.technology.variables.capacity import Capacity
from zen_garden.model.component_types.constraint import GenericConstraint


class TechnologyLifetimeConstraint(GenericConstraint):
    """Define end-of-year capacity from existing and surviving vintages."""

    name = "constraint_technology_lifetime"
    indices = ("set_technologies", "set_location", "set_years")
    masks = (CapacityAvailableMask,)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct define end-of-year capacity from existing and surviving vintages.

        Summary:
            Define end-of-year capacity from existing and surviving vintages.

        Formulation:

        .. math::

            K_{h,q,p,y} =k^{existing}_{h,q,p,y} + \\sum_{v \\in \\mathcal
            V^{active}_{h,y}} \\Delta K_{h,q,p,v}

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
        lhs = lp.merge(
            [
                1 * optimization_model.variables[Capacity],
                -optimization_model.expressions[SurvivingCapacityAdditions],
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = xr.align(
            lhs.const,
            optimization_model.parameters[ExistingCapacities],
            join="left",
        )[1]
        return lhs == rhs
