from zen_garden.topology.generic_variable import GenericVariable


class CostShedDemand(GenericVariable):
    """Variable for cost of shedding demand of carrier"""

    name = "cost_shed_demand"
    indices = ("set_carriers", "set_nodes", "set_time_steps_operation")
    doc = "Variable for cost of shedding demand of carrier"
    unit_category = {"money": 1, "time": -1}

