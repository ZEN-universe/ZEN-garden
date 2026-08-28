from zen_garden.model.component_types.parameter import GenericParameter


class CapacityLimit(GenericParameter):
    """Capacity limit of technologies."""

    name = "capacity_limit"
    indices = ("set_technologies", "set_capacity_types", "set_location", "set_years")
    doc = "Capacity limit of technologies"
    unit_category = {"energy_quantity": 1, "time": -1}
    capacity_types = True
