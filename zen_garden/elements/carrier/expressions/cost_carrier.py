from zen_garden.elements.carrier.parameters.price_export import PriceExport
from zen_garden.elements.carrier.parameters.price_import import PriceImport
from zen_garden.elements.carrier.variables.flow_export import FlowExport
from zen_garden.elements.carrier.variables.flow_import import FlowImport
from zen_garden.model.component_types.expression import GenericExpression


class CostCarrier(GenericExpression):
    """Hourly net cost of importing and exporting carriers."""

    name = "cost_carrier"
    doc = r"""Hourly carrier-trading cost.

    .. math::

        \pi^{\mathrm{imp}}_{c,n,t}F^{\mathrm{imp}}_{c,n,t}
        -\pi^{\mathrm{exp}}_{c,n,t}F^{\mathrm{exp}}_{c,n,t}
    """
    unit_category = {"money": 1, "time": -1}
    indices = ("set_carriers", "set_nodes", "set_years", "set_representative_hours")

    @classmethod
    def get_expression(cls, model_constructor):
        model = model_constructor.optimization_model
        return (
            model.parameters[PriceImport] * model.variables[FlowImport]
            - model.parameters[PriceExport] * model.variables[FlowExport]
        )
