from zen_garden.model.component_types.parameter import GenericParameter


class CapacityAdditionMin(GenericParameter):
    """Parameter which specifies the minimum capacity addition that can be installed."""

    name = "capacity_addition_min"
    indices = ("set_technologies", "set_capacity_types")
    doc = (
        "Parameter which specifies the minimum capacity addition that can be installed"
    )
    unit_category = {"energy_quantity": 1, "time": -1}
    capacity_types = True
