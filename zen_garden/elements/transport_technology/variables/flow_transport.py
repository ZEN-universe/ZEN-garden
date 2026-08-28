import numpy as np
import xarray as xr

from zen_garden.topology.generic_variable import GenericVariable


class FlowTransport(GenericVariable):
    """Variable for carrier flow through transport technology."""

    name = "flow_transport"
    indices = ["set_transport_technologies", "set_edges", "set_time_steps_operation"]
    doc = "Variable for carrier flow through transport technology on edge i and time t"
    unit_category = {"energy_quantity": 1, "time": -1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        index_values, index_names = index_sets
        tech_arr, edge_arr, time_arr = model_constructor.zen_model.sets.tuple_to_arr(
            index_values, index_names
        )
        time_step_year = xr.DataArray(
            [
                model_constructor.time_steps.convert_time_step_operation2year(time)
                for time in time_arr.data
            ]
        )
        capacity = model_constructor.zen_model.variables["capacity"]
        lower = capacity.lower.loc[tech_arr, "power", edge_arr, time_step_year].data
        upper = capacity.upper.loc[tech_arr, "power", edge_arr, time_step_year].data
        return np.stack([lower, upper], axis=-1)
