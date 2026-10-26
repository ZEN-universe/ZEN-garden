import linopy as lp
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.elements.storage_technology.expressions.annualized_capex_from_existing_capacity_energy import (  # noqa: E501
    AnnualizedCapexFromExistingCapacityEnergy,
)
from zen_garden.elements.storage_technology.expressions.annualized_capex_from_new_investments_energy import (  # noqa: E501
    AnnualizedCapexFromNewInvestmentsEnergy,
)
from zen_garden.elements.storage_technology.masks import EnergyCapexActiveMask
from zen_garden.elements.storage_technology.variables.cost_capex_yearly_energy import (
    CostCapexYearlyEnergy,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class CostCapexYearlyEnergyConstraint(GenericConstraint):
    """Define annualized storage energy-capacity CAPEX."""

    name = "constraint_cost_capex_yearly_energy"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    masks = (EnergyCapexActiveMask,)

    @classmethod
    def get_constraint(cls, model_constructor):
        r"""Construct annualized storage energy-capacity CAPEX.

        Summary:
            Combine annualized modeled and existing energy-capacity CAPEX.

        Formulation:

        .. math::

            C^{capex,ann,E}_{h,n,y}=a_h\left(
            \sum_{v\in\mathcal V^{depr}_{h,y}}
            C^{capex,overnight,E}_{h,n,v}+c^{existing,E}_{h,n,y}\right)

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Annualized storage energy-capacity CAPEX constraint.
        """
        model = model_constructor.optimization_model
        lhs = lp.merge(
            [
                1 * model.variables[CostCapexYearlyEnergy],
                -model.expressions[AnnualizedCapexFromNewInvestmentsEnergy],
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = xr.align(
            lhs.const,
            model.expressions[AnnualizedCapexFromExistingCapacityEnergy],
            join="left",
        )[1]
        return lhs == rhs
