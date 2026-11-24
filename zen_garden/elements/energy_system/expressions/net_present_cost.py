from zen_garden.elements.energy_system.expressions.discounted_annual_system_cost import (  # noqa: E501
    DiscountedAnnualSystemCost,
)
from zen_garden.model.component_types.expression import GenericExpression


class NetPresentCost(GenericExpression):
    """Discounted system cost assigned to each modeled year."""

    name = "net_present_cost"
    doc = r"""Discounted annual system cost, expanded to variables and parameters.

    .. math::

        \left[\sum_{i=0}^{d_y-1}(1+r^{\mathrm{disc}})^{-
        [\Delta y(y-y_0)+i]}\right]\left\{\begin{aligned}
        &\sum_{h,p}a(r^{\mathrm{disc}},d_h)
        \left[k^{\mathrm{existing,capex}}_{h,p,y}
        +\sum_{v\in\mathcal V^{\mathrm{depr}}_{h,y}}
        \kappa^{\mathrm{capex}}_{h,p,v}\Delta K_{h,p,v}\right]\\
        +{}&\sum_{h\in\mathcal H^{stor},n}a(r^{\mathrm{disc}},d_h)
        \left[c^{\mathrm{existing,E}}_{h,n,y}
        +\sum_{v\in\mathcal V^{\mathrm{depr}}_{h,y}}
        \kappa^{\mathrm{capex,E}}_{h,n,v}\Delta E_{h,n,v}\right]\\
        +{}&\sum_{h\in\mathcal H^{conv},n,r}w_r\kappa^{\mathrm{op,var}}_{h,y}
        F^{\mathrm{out}}_{h,c_h^{\mathrm{ref}},n,y,r}
        +\sum_{h\in\mathcal H^{stor},n,r}w_r\kappa^{\mathrm{op,var}}_{h,y}
        (F^{\mathrm{charge}}_{h,n,y,r}+F^{\mathrm{discharge}}_{h,n,y,r})\\
        +{}&\sum_{h\in\mathcal H^{trans},e,r}w_r\kappa^{\mathrm{op,var}}_{h,y}
        F^{\mathrm{trans}}_{h,e,y,r}
        +\sum_{h,p}\kappa^{\mathrm{op,fixed}}_{h,p,y}K_{h,p,y}
        +\sum_{h\in\mathcal H^{stor},n}
        \kappa^{\mathrm{op,fixed,E}}_{h,n,y}E_{h,n,y}\\
        +{}&\sum_{c,n,r}w_r(
        \pi^{\mathrm{imp}}_{c,n,y,r}F^{\mathrm{imp}}_{c,n,y,r}
        -\pi^{\mathrm{exp}}_{c,n,y,r}F^{\mathrm{exp}}_{c,n,y,r}
        +\pi^{\mathrm{shed}}_cF^{\mathrm{shed}}_{c,n,y,r})\\
        +{}&\pi^{\mathrm{CO_2}}_y\left[
        \sum_{h\in\mathcal H^{conv},n,r}w_r\varepsilon^{\mathrm{op}}_h
        F^{\mathrm{out}}_{h,c_h^{\mathrm{ref}},n,y,r}
        +\sum_{h\in\mathcal H^{stor},n,r}w_r\varepsilon^{\mathrm{op}}_h
        (F^{\mathrm{charge}}_{h,n,y,r}+F^{\mathrm{discharge}}_{h,n,y,r})\right.\\
        &\left.\qquad+\sum_{h\in\mathcal H^{trans},e,r}w_r
        \varepsilon^{\mathrm{op}}_hF^{\mathrm{trans}}_{h,e,y,r}
        +\sum_{c,n,r}w_r(
        \varepsilon^{\mathrm{imp}}_{c,y}F^{\mathrm{imp}}_{c,n,y,r}
        -\varepsilon^{\mathrm{exp}}_{c,y}F^{\mathrm{exp}}_{c,n,y,r})\right]\\
        +{}&\mathbb 1_{y=y^{last}}\pi^{\mathrm{CO_2,bud}}
        M^{\mathrm{bud,over}}_y
        +\pi^{\mathrm{CO_2,ann}}M^{\mathrm{ann,over}}_y
        \end{aligned}\right\}.
    """
    unit_category = {"money": 1}
    indices = ("set_years",)
    dependencies = (DiscountedAnnualSystemCost,)

    @classmethod
    def get_expression(cls, model_constructor):
        return model_constructor.optimization_model.expressions[
            DiscountedAnnualSystemCost
        ]
