from zen_garden.model.component_types.parameter import GenericParameter


class CapacityAdditionUnbounded(GenericParameter):
    """Unbounded capacity addition that can be added each year."""

    name = "capacity_addition_unbounded"
    indices = ("set_technologies",)
    doc = "Unbounded capacity addition that can be added each year"
    unit_category = {"energy_quantity": 1, "time": -1}
