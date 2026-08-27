from zen_garden.topology.generic_parameter import GenericParameter


class Demand(GenericParameter):
    """Parameter which specifies the carrier demand."""

    name = "demand"
    indices = ("set_carriers", "set_nodes", "set_hours")
    doc = "Parameter which specifies the carrier demand"
    unit_category = {"energy_quantity": 1, "time": -1}
    time_series = True
