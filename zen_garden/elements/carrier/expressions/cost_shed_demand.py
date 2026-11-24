import numpy as np

from zen_garden.elements.carrier.parameters.price_shed_demand import PriceShedDemand
from zen_garden.elements.carrier.variables.shed_demand import ShedDemand
from zen_garden.model.component_types.expression import GenericExpression


class CostShedDemand(GenericExpression):
    """Hourly cost of demand shedding where a finite price is configured."""

    name = "cost_shed_demand"
    doc = r"""Hourly demand-shedding cost, zero where shedding is disabled.

    .. math::

        \begin{cases}
        \pi^{\mathrm{shed}}_c F^{\mathrm{shed}}_{c,n,t},
        &\pi^{\mathrm{shed}}_c<\infty,\\
        0,&\text{otherwise.}
        \end{cases}
    """
    unit_category = {"money": 1, "time": -1}
    indices = ("set_carriers", "set_nodes", "set_years", "set_representative_hours")

    @classmethod
    def get_expression(cls, model_constructor):
        model = model_constructor.optimization_model
        price = model.parameters[PriceShedDemand]
        return (price * model.variables[ShedDemand]).where(price != np.inf, 0)
