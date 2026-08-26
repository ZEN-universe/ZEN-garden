from zen_garden.topology.generic_variable import GenericVariable


class Capacity(GenericVariable):
    """Variable for installed technology capacity."""

    name = "capacity"
    indices = ["set_technologies", "set_capacity_types", "set_location", "set_years"]
    doc = "Variable for size of installed technology at location l and time t"
    unit_category = {"energy_quantity": 1, "time": -1}

