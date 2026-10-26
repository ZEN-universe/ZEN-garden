"""Discounted annual system cost."""

import pandas as pd

from zen_garden.elements.energy_system.parameters.discount_rate import DiscountRate
from zen_garden.elements.energy_system.parameters.interval_between_years import (
    IntervalBetweenYears,
)
from zen_garden.elements.energy_system.variables.cost_total import CostTotal
from zen_garden.model.component_types.expression import GenericExpression


class DiscountedAnnualSystemCost(GenericExpression):
    """Apply intra-period economic discounting to annual system cost."""

    name = "discounted_annual_system_cost"
    indices = ("set_years",)

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct apply intra-period economic discounting to annual system cost.

        Summary:
            Apply intra-period economic discounting to annual system cost.

        Formulation:

        .. math::

            C^{total}_y\\sum_{i=0}^{d_y-1}(1+q)^{-[\\Delta y(y-y_0)+i]}

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
        years = model_constructor.model_schema.set_years
        factor = pd.Series(index=years, dtype=float)
        for year in years:
            interval = (
                1
                if year == model_constructor.model_schema.set_years_entire_horizon[-1]
                else optimization_model.parameters[IntervalBetweenYears]
            )
            factor[year] = sum(
                (1 / (1 + optimization_model.parameters[DiscountRate]))
                ** (
                    optimization_model.parameters[IntervalBetweenYears]
                    * (year - years[0])
                    + intermediate_step
                )
                for intermediate_step in range(interval)
            )
        return optimization_model.variables[CostTotal] * factor
