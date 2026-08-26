from zen_garden.topology.generic_variable import GenericVariable


class ShedDemand(GenericVariable):
    """Variable for shedding carrier demand."""

    name = "shed_demand"
    indices = ("set_carriers", "set_nodes", "set_time_steps_operation")
    doc = "Variable for shedding demand of carrier"
    unit_category = {"energy_quantity": 1, "time": -1}

