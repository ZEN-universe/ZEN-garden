import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.elements.technology.expressions.annualized_capex_from_existing_capacity import (  # noqa: E501
    AnnualizedCapexFromExistingCapacity,
)
from zen_garden.elements.technology.expressions.annualized_capex_from_new_investments import (  # noqa: E501
    AnnualizedCapexFromNewInvestments,
)
from zen_garden.elements.technology.masks import CapexActiveMask
from zen_garden.model.component_types.expression import GenericExpression


class CostCapexYearly(GenericExpression):
    """Annualized CAPEX from existing and modeled capacity."""

    name = "cost_capex_yearly"
    doc = r"""Annualized power-capacity investment cost.

    .. math::

        a(r^{\mathrm{disc}},d_h)\left[
        k^{\mathrm{existing,capex}}_{h,p,y}
        +\sum_{v\in\mathcal V^{\mathrm{depr}}_{h,y}}
        \underbrace{\kappa^{\mathrm{capex}}_{h,p,v}\Delta K_{h,p,v}
        +\mathbb 1_{h\in\mathcal H^{trans}}
        \kappa^{\mathrm{distance}}_{h,p,v}G_{h,p,v}}
        _{\text{applicable conversion, storage, or transport investment}}
        \right]
    """
    unit_category = {"money": 1}
    indices = ("set_technologies", "set_location", "set_years")
    dependencies = (
        AnnualizedCapexFromNewInvestments,
        AnnualizedCapexFromExistingCapacity,
    )
    masks = (CapexActiveMask,)
    mask_default_value = 0.0

    @classmethod
    def get_expression(cls, model_constructor):
        model = model_constructor.optimization_model
        expression = model.expressions[AnnualizedCapexFromNewInvestments]
        data = xr.Dataset(
            {
                "coeffs": expression.coeffs,
                "vars": expression.vars,
                "const": expression.const
                + model.expressions[AnnualizedCapexFromExistingCapacity],
            }
        )
        return LinearExpression(data, model.lp_model)
