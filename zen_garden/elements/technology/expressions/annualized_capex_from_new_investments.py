"""Annualized CAPEX attributable to modeled investments."""

import itertools

import pandas as pd
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.elements.energy_system.parameters.discount_rate import DiscountRate
from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.technology.expressions.cost_capex_overnight import (
    CostCapexOvernight,
)
from zen_garden.elements.technology.lifetime import get_active_vintage_years
from zen_garden.elements.technology.parameters.depreciation_time import DepreciationTime
from zen_garden.elements.technology.variables.capacity import Capacity
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.linear_expression import transpose_linear_expression


class AnnualizedCapexFromNewInvestments(GenericExpression):
    """Annualize active overnight-CAPEX vintages."""

    name = "annualized_capex_from_new_investments"
    unit_category = {"money": 1}
    indices = ("set_technologies", "set_location", "set_years")
    dependencies = (CostCapexOvernight,)

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct annualize active overnight-capex vintages.

        Summary:
            Annualize active overnight-CAPEX vintages.

        Formulation:

        .. math::

            \\sum_{v\\in\\mathcal V^{depr}_{h,y}}a(q,d_h)C^{capex,overnight}_{h,q,p,v}

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
        yearly_capex = optimization_model.variables[Capacity]
        discount_rate = optimization_model.parameters[DiscountRate]
        depreciation_time = optimization_model.parameters[DepreciationTime]
        annuity = (
            ((1 + discount_rate) ** depreciation_time * discount_rate)
            / ((1 + discount_rate) ** depreciation_time - 1)
            if discount_rate != 0
            else 1 / depreciation_time
        )
        vintages = pd.MultiIndex.from_tuples(
            [
                (technology, year, vintage)
                for technology, year in itertools.product(
                    optimization_model.sets[SetTechnologies],
                    optimization_model.sets[SetYears],
                )
                for vintage in get_active_vintage_years(
                    model_constructor,
                    technology,
                    year,
                    use_depreciation_time=True,
                )
            ],
            names=["set_technologies", "set_years", "set_years_prev"],
        )
        active = pd.Series(1, index=vintages).to_xarray()
        active = active.broadcast_like(yearly_capex.lower).fillna(0)
        active = active.transpose(*yearly_capex.dims, "set_years_prev")
        capex = optimization_model.expressions[CostCapexOvernight].rename(
            {"set_years": "set_years_prev"}
        )
        capex = transpose_linear_expression(
            1 * capex.broadcast_like(active), active.dims
        )
        weighted = active * annuity * capex
        if "_term" in weighted.const.dims:
            weighted = LinearExpression(
                xr.Dataset(
                    {
                        "coeffs": weighted.coeffs,
                        "vars": weighted.vars,
                        "const": weighted.const.sum("_term"),
                    }
                ),
                weighted.model,
            )
        return weighted.sum("set_years_prev")
