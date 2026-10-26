import numpy as np

from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.elements.transport_technology.masks import FiniteTransportRouteMask
from zen_garden.model.component_types.variable import GenericVariable


class FlowTransportLoss(GenericVariable):
    """Variable for carrier flow loss through transport technology."""

    name = "flow_transport_loss"
    symbol = r"F^{\mathrm{loss}}_{h,e,t}"
    indices = [
        "set_transport_technologies",
        "set_edges",
        "set_years",
        "set_representative_hours",
    ]
    doc = "Carrier flow through a transport technology."
    unit_category = {"energy_quantity": 1, "time": -1}
    masks = [PowerOperationMask, FiniteTransportRouteMask]
    mask_default_value = 0.0

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return 0, np.inf
