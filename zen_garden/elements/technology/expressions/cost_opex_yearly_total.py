from zen_garden.elements.technology.expressions.cost_opex_yearly import CostOpexYearly
from zen_garden.model.component_types.expression import GenericExpression


class CostOpexYearlyTotal(GenericExpression):
    """Total annual operating cost across technologies and locations."""

    name = "cost_opex_yearly_total"
    doc = r"""Total annual operating cost.

    .. math::

        \begin{aligned}
        &\sum_{h\in\mathcal H^{conv},n,r}w_r\kappa^{\mathrm{op,var}}_{h,y}
        F^{\mathrm{out}}_{h,c_h^{\mathrm{ref}},n,y,r}\\
        +{}&\sum_{h\in\mathcal H^{stor},n,r}w_r\kappa^{\mathrm{op,var}}_{h,y}
        (F^{\mathrm{charge}}_{h,n,y,r}+F^{\mathrm{discharge}}_{h,n,y,r})\\
        +{}&\sum_{h\in\mathcal H^{trans},e,r}w_r\kappa^{\mathrm{op,var}}_{h,y}
        F^{\mathrm{trans}}_{h,e,y,r}
        +\sum_{h,p}\kappa^{\mathrm{op,fixed}}_{h,p,y}K_{h,p,y}\\
        +{}&\sum_{h\in\mathcal H^{stor},n}
        \kappa^{\mathrm{op,fixed,E}}_{h,n,y}E_{h,n,y}.
        \end{aligned}
    """
    unit_category = {"money": 1}
    indices = ("set_years",)
    dependencies = (CostOpexYearly,)

    @classmethod
    def get_expression(cls, model_constructor):
        return model_constructor.optimization_model.expressions[CostOpexYearly].sum(
            ["set_technologies", "set_location"]
        )
