import numpy as np
import xarray as xr

from zen_garden.model.component_types.variable import GenericVariable


class FlowStorageCharge(GenericVariable):
    """Variable for carrier flow into storage technology."""

    name = "flow_storage_charge"
    indices = ["set_storage_technologies", "set_nodes", "set_time_steps_operation"]
    doc = "Variable for carrier flow into storage technology on node i and time t"
    unit_category = {"energy_quantity": 1, "time": -1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        index_values, index_names = index_sets
        tech_arr, node_arr, time_arr = (
            model_constructor.optimization_model.sets.tuple_to_arr(
                index_values, index_names
            )
        )
        time_step_year = xr.DataArray(
            [
                model_constructor.time_steps.convert_time_step_operation2year(time)
                for time in time_arr.data
            ]
        )
        capacity = model_constructor.optimization_model.variables["capacity"]
        lower = capacity.lower.loc[tech_arr, "power", node_arr, time_step_year].data
        upper = capacity.upper.loc[tech_arr, "power", node_arr, time_step_year].data
        return np.stack([lower, upper], axis=-1)
