from zen_garden.elements.storage_technology.expressions.cost_capex_yearly_energy import (  # noqa: E501
    CostCapexYearlyEnergy,
)
from zen_garden.elements.technology.expressions.cost_capex_yearly import CostCapexYearly
from zen_garden.model.component_types.expression import GenericExpression


class CostCapexYearlyTotal(GenericExpression):
    """Total annual power- and energy-capacity CAPEX."""

    name = "cost_capex_yearly_total"
    doc = r"""Total annualized power- and storage-energy CAPEX.

    .. math::

        \sum_{h,p}a(r^{\mathrm{disc}},d_h)\left[
        k^{\mathrm{existing,capex}}_{h,p,y}
        +\sum_{v\in\mathcal V^{\mathrm{depr}}_{h,y}}
        \kappa^{\mathrm{capex}}_{h,p,v}\Delta K_{h,p,v}\right]
        +\sum_{h\in\mathcal H^{stor},n}a(r^{\mathrm{disc}},d_h)\left[
        c^{\mathrm{existing,E}}_{h,n,y}
        +\sum_{v\in\mathcal V^{\mathrm{depr}}_{h,y}}
        \kappa^{\mathrm{capex,E}}_{h,n,v}\Delta E_{h,n,v}\right]
    """
    unit_category = {"money": 1}
    indices = ("set_years",)
    dependencies = (CostCapexYearly, CostCapexYearlyEnergy)

    @classmethod
    def get_expression(cls, model_constructor):
        model = model_constructor.optimization_model
        total = model.expressions[CostCapexYearly].sum(
            ["set_technologies", "set_location"]
        )
        if CostCapexYearlyEnergy in model.expressions:
            total += model.expressions[CostCapexYearlyEnergy].sum(
                ["set_storage_technologies", "set_nodes"]
            )
        return total
