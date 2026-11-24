from zen_garden.elements.energy_system.expressions.annual_carbon_cost import (
    AnnualCarbonCost,
)
from zen_garden.model.component_types.expression import GenericExpression


class CostCarbonEmissionsTotal(GenericExpression):
    """Annual carbon-emissions and overshoot cost."""

    name = "cost_carbon_emissions_total"
    doc = r"""Annual emissions and overshoot cost.

    .. math::

        \begin{aligned}
        \pi^{\mathrm{CO_2}}_y\Bigg[&
        \sum_{h\in\mathcal H^{conv},n,r}w_r\varepsilon^{\mathrm{op}}_h
        F^{\mathrm{out}}_{h,c_h^{\mathrm{ref}},n,y,r}\\
        &+\sum_{h\in\mathcal H^{stor},n,r}w_r\varepsilon^{\mathrm{op}}_h
        (F^{\mathrm{charge}}_{h,n,y,r}+F^{\mathrm{discharge}}_{h,n,y,r})\\
        &+\sum_{h\in\mathcal H^{trans},e,r}w_r\varepsilon^{\mathrm{op}}_h
        F^{\mathrm{trans}}_{h,e,y,r}\\
        &+\sum_{c,n,r}w_r(
        \varepsilon^{\mathrm{imp}}_{c,y}F^{\mathrm{imp}}_{c,n,y,r}
        -\varepsilon^{\mathrm{exp}}_{c,y}F^{\mathrm{exp}}_{c,n,y,r})\Bigg]\\
        &+\mathbb 1_{y=y^{last}}\pi^{\mathrm{CO_2,bud}}
        M^{\mathrm{bud,over}}_y
        +\pi^{\mathrm{CO_2,ann}}M^{\mathrm{ann,over}}_y.
        \end{aligned}
    """
    unit_category = {"money": 1}
    indices = ("set_years",)
    dependencies = (AnnualCarbonCost,)

    @classmethod
    def get_expression(cls, model_constructor):
        return model_constructor.optimization_model.expressions[AnnualCarbonCost]
