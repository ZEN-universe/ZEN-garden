import numpy as np

from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.elements.technology.variables.capacity import Capacity
from zen_garden.elements.transport_technology.masks import FiniteTransportRouteMask
from zen_garden.model.component_types.variable import GenericVariable


class FlowTransport(GenericVariable):
    """Variable for carrier flow through transport technology."""

    name = "flow_transport"
    symbol = r"F^{\mathrm{trans}}_{h,e,t}"
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
        index_values, index_names = index_sets
        tech_arr, edge_arr, year_arr, _ = (
            model_constructor.optimization_model.sets.tuple_to_arr(
                index_values, index_names
            )
        )
        capacity = model_constructor.optimization_model.variables[Capacity]
        lower = capacity.lower.loc[tech_arr, edge_arr, year_arr].data
        upper = capacity.upper.loc[tech_arr, edge_arr, year_arr].data
        return np.stack([lower, upper], axis=-1)
