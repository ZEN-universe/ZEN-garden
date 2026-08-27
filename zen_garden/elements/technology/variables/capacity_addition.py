import numpy as np

from zen_garden.topology.generic_variable import GenericVariable


class CapacityAddition(GenericVariable):
    """Variable for built/added technology capacity."""

    name = "capacity_addition"
    indices = ["set_technologies", "set_capacity_types", "set_location", "set_years"]
    doc = (
        "Variable for size of built technology (invested capacity after construction) "
        "at location l and time t"
    )
    unit_category = {"energy_quantity": 1, "time": -1}

    @classmethod
    def get_bounds(cls):
        return 0, np.inf
