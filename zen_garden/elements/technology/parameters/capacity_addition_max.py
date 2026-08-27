from zen_garden.topology.generic_parameter import GenericParameter


class CapacityAdditionMax(GenericParameter):
    """Parameter which specifies the maximum capacity addition that can be installed."""

    name = "capacity_addition_max"
    indices = ("set_technologies", "set_capacity_types")
    doc = (
        "Parameter which specifies the maximum capacity addition that can be installed"
    )
    unit_category = {"energy_quantity": 1, "time": -1}
    capacity_types = True
