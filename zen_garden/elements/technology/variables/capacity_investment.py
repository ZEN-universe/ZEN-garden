from zen_garden.topology.generic_variable import GenericVariable


class CapacityInvestment(GenericVariable):
    """Variable for invested technology capacity."""

    name = "capacity_investment"
    indices = ("set_technologies", "set_capacity_types", "set_location", "set_years")
    doc = "Variable for size of invested technology at location l and time t"
    unit_category = {"energy_quantity": 1, "time": -1}

