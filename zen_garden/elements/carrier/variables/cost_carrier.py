from zen_garden.topology.generic_variable import GenericVariable


class CostCarrier(GenericVariable):
    """Variable for carrier import/export cost."""

    name = "cost_carrier"
    indices = ("set_carriers", "set_nodes", "set_time_steps_operation")
    doc = "Variable for node- and time-dependent carrier cost due to import and export"
    unit_category = {"money": 1, "time": -1}

