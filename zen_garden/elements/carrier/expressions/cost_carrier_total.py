from zen_garden.elements.carrier.expressions.annual_carrier_cost import (
    AnnualCarrierCost,
)
from zen_garden.model.component_types.expression import GenericExpression


class CostCarrierTotal(GenericExpression):
    """Annual carrier trading and demand-shedding cost."""

    name = "cost_carrier_total"
    doc = r"""Annual carrier-trading and demand-shedding cost.

    .. math::

        \sum_{c,n,r} w_r\left(
        \pi^{\mathrm{imp}}_{c,n,y,r}F^{\mathrm{imp}}_{c,n,y,r}
        -\pi^{\mathrm{exp}}_{c,n,y,r}F^{\mathrm{exp}}_{c,n,y,r}
        +\pi^{\mathrm{shed}}_cF^{\mathrm{shed}}_{c,n,y,r}
        \right)
    """
    unit_category = {"money": 1}
    indices = ("set_years",)
    dependencies = (AnnualCarrierCost,)

    @classmethod
    def get_expression(cls, model_constructor):
        return model_constructor.optimization_model.expressions[AnnualCarrierCost]
