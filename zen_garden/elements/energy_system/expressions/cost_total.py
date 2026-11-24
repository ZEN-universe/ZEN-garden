from zen_garden.elements.carrier.expressions.cost_carrier_total import CostCarrierTotal
from zen_garden.elements.energy_system.expressions.cost_carbon_emissions_total import (
    CostCarbonEmissionsTotal,
)
from zen_garden.elements.technology.expressions.cost_capex_yearly_total import (
    CostCapexYearlyTotal,
)
from zen_garden.elements.technology.expressions.cost_opex_yearly_total import (
    CostOpexYearlyTotal,
)
from zen_garden.model.component_types.expression import GenericExpression


class CostTotal(GenericExpression):
    """Total annual system cost from CAPEX, OPEX, carriers, and emissions."""

    name = "cost_total"
    doc = r"""Total annual system cost, fully expanded to variables and parameters.

    .. math::

        \begin{aligned}
        &\underbrace{\sum_{h,p}a(r^{\mathrm{disc}},d_h)
        \left[k^{\mathrm{existing,capex}}_{h,p,y}
        +\sum_{v\in\mathcal V^{\mathrm{depr}}_{h,y}}
        \kappa^{\mathrm{capex}}_{h,p,v}\Delta K_{h,p,v}\right]
        +\sum_{h\in\mathcal H^{stor},n}a(r^{\mathrm{disc}},d_h)
        \left[c^{\mathrm{existing,E}}_{h,n,y}
        +\sum_{v\in\mathcal V^{\mathrm{depr}}_{h,y}}
        \kappa^{\mathrm{capex,E}}_{h,n,v}\Delta E_{h,n,v}\right]}
        _{\text{annualized investment cost}}\\
        +{}&\underbrace{\sum_{h\in\mathcal H^{conv},n,r}
        w_r\kappa^{\mathrm{op,var}}_{h,y}
        F^{\mathrm{out}}_{h,c_h^{\mathrm{ref}},n,y,r}
        +\sum_{h\in\mathcal H^{stor},n,r}w_r\kappa^{\mathrm{op,var}}_{h,y}
        (F^{\mathrm{charge}}_{h,n,y,r}+F^{\mathrm{discharge}}_{h,n,y,r})
        +\sum_{h\in\mathcal H^{trans},e,r}w_r\kappa^{\mathrm{op,var}}_{h,y}
        F^{\mathrm{trans}}_{h,e,y,r}
        +\sum_{h,p}\kappa^{\mathrm{op,fixed}}_{h,p,y}K_{h,p,y}
        +\sum_{h\in\mathcal H^{stor},n}
        \kappa^{\mathrm{op,fixed,E}}_{h,n,y}E_{h,n,y}}
        _{\text{variable and fixed operating cost}}\\
        +{}&\underbrace{\sum_{c,n,r}w_r(
        \pi^{\mathrm{imp}}_{c,n,y,r}F^{\mathrm{imp}}_{c,n,y,r}
        -\pi^{\mathrm{exp}}_{c,n,y,r}F^{\mathrm{exp}}_{c,n,y,r}
        +\pi^{\mathrm{shed}}_cF^{\mathrm{shed}}_{c,n,y,r})}
        _{\text{carrier and shedding cost}}\\
        +{}&\underbrace{\pi^{\mathrm{CO_2}}_y\Big[
        \sum_{h\in\mathcal H^{conv},n,r}w_r\varepsilon^{\mathrm{op}}_h
        F^{\mathrm{out}}_{h,c_h^{\mathrm{ref}},n,y,r}
        +\sum_{h\in\mathcal H^{stor},n,r}w_r\varepsilon^{\mathrm{op}}_h
        (F^{\mathrm{charge}}_{h,n,y,r}+F^{\mathrm{discharge}}_{h,n,y,r})
        +\sum_{h\in\mathcal H^{trans},e,r}w_r\varepsilon^{\mathrm{op}}_h
        F^{\mathrm{trans}}_{h,e,y,r}
        +\sum_{c,n,r}w_r(
        \varepsilon^{\mathrm{imp}}_{c,y}F^{\mathrm{imp}}_{c,n,y,r}
        -\varepsilon^{\mathrm{exp}}_{c,y}F^{\mathrm{exp}}_{c,n,y,r})\Big]
        +\mathbb 1_{y=y^{last}}\pi^{\mathrm{CO_2,bud}}M^{\mathrm{bud,over}}_y
        +\pi^{\mathrm{CO_2,ann}}M^{\mathrm{ann,over}}_y}
        _{\text{emissions cost}}.
        \end{aligned}
    """
    unit_category = {"money": 1}
    indices = ("set_years",)
    dependencies = (
        CostCapexYearlyTotal,
        CostOpexYearlyTotal,
        CostCarrierTotal,
        CostCarbonEmissionsTotal,
    )

    @classmethod
    def get_expression(cls, model_constructor):
        model = model_constructor.optimization_model
        return (
            model.expressions[CostCapexYearlyTotal]
            + model.expressions[CostOpexYearlyTotal]
            + model.expressions[CostCarrierTotal]
            + model.expressions[CostCarbonEmissionsTotal]
        )
