"""Annualized CAPEX from modeled storage energy investments."""

import itertools

import pandas as pd

from zen_garden.elements.energy_system.parameters.discount_rate import DiscountRate
from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.storage_technology.variables.cost_capex_overnight_energy import (  # noqa: E501
    CostCapexOvernightEnergy,
)
from zen_garden.elements.storage_technology.variables.cost_capex_yearly_energy import (
    CostCapexYearlyEnergy,
)
from zen_garden.elements.technology.lifetime import get_active_vintage_years
from zen_garden.elements.technology.parameters.depreciation_time import DepreciationTime
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.linear_expression import transpose_linear_expression


class AnnualizedCapexFromNewInvestmentsEnergy(GenericExpression):
    """Annualize active storage energy-capacity investment vintages."""

    name = "annualized_capex_from_new_investments_energy"
    indices = ("set_storage_technologies", "set_nodes", "set_years")

    @classmethod
    def get_expression(cls, model_constructor):
        r"""Construct annualized modeled storage energy CAPEX.

        Summary:
            Annualize active storage energy-capacity investment vintages.

        Formulation:

        .. math::

            \sum_{v\in\mathcal V^{depr}_{h,y}}
            a(r,d_h)C^{capex,overnight,E}_{h,n,v}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Annualized modeled storage energy-capacity CAPEX.
        """
        model = model_constructor.optimization_model
        yearly_capex = model.variables[CostCapexYearlyEnergy]
        discount_rate = model.parameters[DiscountRate]
        depreciation_time = model.parameters[DepreciationTime]
        annuity = (
            ((1 + discount_rate) ** depreciation_time * discount_rate)
            / ((1 + discount_rate) ** depreciation_time - 1)
            if discount_rate != 0
            else 1 / depreciation_time
        )
        storage_technologies = model.sets[SetStorageTechnologies]
        annuity = annuity.sel(set_technologies=storage_technologies).rename(
            set_technologies="set_storage_technologies"
        )
        vintages = pd.MultiIndex.from_tuples(
            [
                (technology, year, vintage)
                for technology, year in itertools.product(
                    storage_technologies, model.sets[SetYears]
                )
                for vintage in get_active_vintage_years(
                    model_constructor,
                    technology,
                    year,
                    use_depreciation_time=True,
                )
            ],
            names=["set_storage_technologies", "set_years", "set_years_prev"],
        )
        active = pd.Series(1, index=vintages).to_xarray()
        active = active.broadcast_like(yearly_capex.lower).fillna(0)
        active = active.transpose(*yearly_capex.dims, "set_years_prev")
        capex = model.variables[CostCapexOvernightEnergy].rename(
            {"set_years": "set_years_prev"}
        )
        capex = transpose_linear_expression(
            1 * capex.broadcast_like(active), active.dims
        )
        return (active * annuity * capex).sum("set_years_prev")
