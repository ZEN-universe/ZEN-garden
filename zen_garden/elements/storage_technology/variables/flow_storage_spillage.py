import numpy as np

from zen_garden.topology.generic_variable import GenericVariable


class FlowStorageSpillage(GenericVariable):
    """Variable for storage spillage."""

    name = "flow_storage_spillage"
    indices = ["set_storage_technologies", "set_nodes", "set_time_steps_operation"]
    doc = "Variable for storage spillage of storage technology on node i in each storage time step"
    unit_category = {"energy_quantity": 1, "time": -1}

    @classmethod
    def get_bounds(cls):
        return 0, np.inf
