import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.elements.storage_technology.expressions.annualized_capex_from_existing_capacity_energy import (  # noqa: E501
    AnnualizedCapexFromExistingCapacityEnergy,
)
from zen_garden.elements.storage_technology.expressions.annualized_capex_from_new_investments_energy import (  # noqa: E501
    AnnualizedCapexFromNewInvestmentsEnergy,
)
from zen_garden.elements.storage_technology.masks import EnergyCapexActiveMask
from zen_garden.model.component_types.expression import GenericExpression


class CostCapexYearlyEnergy(GenericExpression):
    """Annualized storage energy-capacity CAPEX."""

    name = "cost_capex_yearly_energy"
    doc = r"""Annualized storage energy-capacity investment cost.

    .. math::

        a(r^{\mathrm{disc}},d_h)\left[
        c^{\mathrm{existing,E}}_{h,n,y}
        +\sum_{v\in\mathcal V^{\mathrm{depr}}_{h,y}}
        \kappa^{\mathrm{capex,E}}_{h,n,v}\Delta E_{h,n,v}\right]
    """
    unit_category = {"money": 1}
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    dependencies = (
        AnnualizedCapexFromNewInvestmentsEnergy,
        AnnualizedCapexFromExistingCapacityEnergy,
    )
    masks = (EnergyCapexActiveMask,)
    mask_default_value = 0.0

    @classmethod
    def get_expression(cls, model_constructor):
        model = model_constructor.optimization_model
        expression = model.expressions[AnnualizedCapexFromNewInvestmentsEnergy]
        data = xr.Dataset(
            {
                "coeffs": expression.coeffs,
                "vars": expression.vars,
                "const": expression.const
                + model.expressions[AnnualizedCapexFromExistingCapacityEnergy],
            }
        )
        return LinearExpression(data, model.lp_model)
