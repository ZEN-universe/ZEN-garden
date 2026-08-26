import numpy as np
from zen_garden.topology.generic_variable import GenericVariable


class StorageLevel(GenericVariable):
    """Variable for storage level."""

    name = "storage_level"
    indices = ["set_storage_technologies", "set_nodes", "set_time_steps_storage"]
    doc = "Variable for storage level of storage technology on node in each storage time step"
    unit_category = {"energy_quantity": 1}

    @classmethod
    def get_bounds(cls):
        return 0, np.inf