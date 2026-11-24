from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.elements.transport_technology.masks import FiniteTransportRouteMask
from zen_garden.elements.transport_technology.parameters.transport_loss_factor import (
    TransportLossFactor,
)
from zen_garden.elements.transport_technology.variables.flow_transport import (
    FlowTransport,
)
from zen_garden.model.component_types.expression import GenericExpression


class FlowTransportLoss(GenericExpression):
    """Carrier flow lost along finite transport routes."""

    name = "flow_transport_loss"
    doc = r"""Hourly carrier flow lost along a transport edge.

    .. math::

        \lambda^{\mathrm{trans}}_{h,e}F^{\mathrm{trans}}_{h,e,t}
    """
    unit_category = {"energy_quantity": 1, "time": -1}
    indices = (
        "set_transport_technologies",
        "set_edges",
        "set_years",
        "set_representative_hours",
    )
    masks = (PowerOperationMask, FiniteTransportRouteMask)
    mask_default_value = 0.0

    @classmethod
    def get_expression(cls, model_constructor):
        model = model_constructor.optimization_model
        return model.parameters[TransportLossFactor] * model.variables[FlowTransport]
