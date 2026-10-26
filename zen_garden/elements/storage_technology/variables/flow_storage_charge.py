import numpy as np

from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.elements.technology.variables.capacity import Capacity
from zen_garden.model.component_types.variable import GenericVariable


class FlowStorageCharge(GenericVariable):
    """Variable for carrier flow into storage technology."""

    name = "flow_storage_charge"
    symbol = r"F^{\mathrm{ch}}_{h,n,t}"
    indices = [
        "set_storage_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    ]
    doc = "Carrier flow into a storage technology."
    unit_category = {"energy_quantity": 1, "time": -1}
    masks = [PowerOperationMask]
    mask_default_value = 0.0

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        index_values, index_names = index_sets
        tech_arr, node_arr, year_arr, _ = (
            model_constructor.optimization_model.sets.tuple_to_arr(
                index_values, index_names
            )
        )
        capacity = model_constructor.optimization_model.variables[Capacity]
        lower = capacity.lower.loc[tech_arr, node_arr, year_arr].data
        upper = capacity.upper.loc[tech_arr, node_arr, year_arr].data
        return np.stack([lower, upper], axis=-1)
