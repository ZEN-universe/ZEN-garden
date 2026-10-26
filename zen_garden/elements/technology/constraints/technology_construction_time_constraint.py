"""Capacity commissioning constraint."""

import linopy as lp
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.elements.technology.expressions.capacity_additions_from_new_investments import (  # noqa: E501
    CapacityAdditionsFromNewInvestments,
)
from zen_garden.elements.technology.expressions.capacity_additions_from_pre_horizon_investments import (  # noqa: E501
    CapacityAdditionsFromPreHorizonInvestments,
)
from zen_garden.elements.technology.masks import CapacityAdditionEligibleMask
from zen_garden.elements.technology.variables.capacity_addition import CapacityAddition
from zen_garden.model.component_types.constraint import GenericConstraint


class TechnologyConstructionTimeConstraint(GenericConstraint):
    """Link capacity additions to investments using construction lead time."""

    name = "constraint_technology_construction_time"
    indices = ("set_technologies", "set_location", "set_years")
    masks = (CapacityAdditionEligibleMask,)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct the model component.

        Summary:
            Link capacity additions to investments using construction lead time.

        Formulation:

        .. math::

             \\Delta K_{h,q,p,y} = \\sum_{v:v +\\tau_h =y}K^{investment}_{h,q,p,v} +
            \\sum_{v\\notin \\mathcal Y:v +\\tau_h =y}k^{investment,existing}_{h,q,p,v}

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
                1 * optimization_model.variables[CapacityAddition],
                -optimization_model.expressions[CapacityAdditionsFromNewInvestments],
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = xr.align(
            lhs.const,
            optimization_model.expressions[CapacityAdditionsFromPreHorizonInvestments],
            join="left",
        )[1]
        return lhs == rhs
