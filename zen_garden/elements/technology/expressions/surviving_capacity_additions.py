"""Capacity additions that remain within their technical lifetime."""

import itertools

import pandas as pd

from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.technology.lifetime import get_active_vintage_years
from zen_garden.elements.technology.variables.capacity import Capacity
from zen_garden.elements.technology.variables.capacity_addition import CapacityAddition
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.linear_expression import transpose_linear_expression


class SurvivingCapacityAdditions(GenericExpression):
    """Sum modeled capacity vintages that have not retired."""

    name = "surviving_capacity_additions"
    unit_category = {"energy_quantity": 1, "time": -1}
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct sum modeled capacity vintages that have not retired.

        Summary:
            Sum modeled capacity vintages that have not retired.

        Formulation:

        .. math::

            \\sum_{v\\in\\mathcal V^{active}_{h,y}}\\Delta K_{h,q,p,v}

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Expression over its declared model indices.
        """
        optimization_model = model_constructor.optimization_model
        capacity = optimization_model.variables[Capacity]
        active_vintages = pd.MultiIndex.from_tuples(
            [
                (technology, year, vintage)
                for technology, year in itertools.product(
                    optimization_model.sets[SetTechnologies],
                    optimization_model.sets[SetYears],
                )
                for vintage in get_active_vintage_years(
                    model_constructor, technology, year
                )
            ],
            names=["set_technologies", "set_years", "set_years_prev"],
        )
        active = pd.Series(1, index=active_vintages).to_xarray()
        active = active.broadcast_like(capacity.lower).fillna(0)
        active = active.transpose(*capacity.dims, "set_years_prev")
        additions = optimization_model.variables[CapacityAddition].rename(
            {"set_years": "set_years_prev"}
        )
        additions = transpose_linear_expression(
            1 * additions.broadcast_like(active), active.dims
        )
        return (active * additions).sum("set_years_prev")
