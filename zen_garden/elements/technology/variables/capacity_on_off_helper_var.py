import numpy as np

from zen_garden.topology.generic_variable import GenericVariable


class CapacityOnOffHelperVar(GenericVariable):
    """Variable for capacity on/off helper."""

    name = "capacity_on_off_helper_var"
    indices = ["set_technologies", "set_location", "set_time_steps_operation"]
    doc = "Variable substituting the product of capacity and tech_on_var"
    unit_category = {"energy_quantity": 1, "time": -1}

    @classmethod
    def get_bounds(cls):
        return 0.0, np.inf
