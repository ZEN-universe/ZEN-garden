from zen_garden.elements.technology.expressions.annual_fixed_opex import AnnualFixedOpex
from zen_garden.elements.technology.expressions.annual_variable_opex import (
    AnnualVariableOpex,
)
from zen_garden.elements.technology.masks import CapacityAnyTypeMask
from zen_garden.model.component_types.expression import GenericExpression


class CostOpexYearly(GenericExpression):
    """Annual fixed and variable operating cost by technology and location."""

    name = "cost_opex_yearly"
    doc = r"""Annual fixed and variable operating cost.

    .. math::

        \sum_r w_r\kappa^{\mathrm{op,var}}_{h,y}
        \begin{cases}
        F^{\mathrm{out}}_{h,c_h^{\mathrm{ref}},n,y,r},&h\in\mathcal H^{conv},\\
        F^{\mathrm{charge}}_{h,n,y,r}+F^{\mathrm{discharge}}_{h,n,y,r},
        &h\in\mathcal H^{stor},\\
        F^{\mathrm{trans}}_{h,e,y,r},&h\in\mathcal H^{trans}
        \end{cases}
        +\kappa^{\mathrm{op,fixed}}_{h,p,y}K_{h,p,y}
        +\mathbb 1_{h\in\mathcal H^{stor}}
        \kappa^{\mathrm{op,fixed,E}}_{h,n,y}E_{h,n,y}
    """
    unit_category = {"money": 1}
    indices = ("set_technologies", "set_location", "set_years")
    dependencies = (AnnualVariableOpex, AnnualFixedOpex)
    masks = (CapacityAnyTypeMask,)
    mask_default_value = 0.0

    @classmethod
    def get_expression(cls, model_constructor):
        model = model_constructor.optimization_model
        return (
            model.expressions[AnnualVariableOpex] + model.expressions[AnnualFixedOpex]
        )
