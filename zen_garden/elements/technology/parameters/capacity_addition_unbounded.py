from zen_garden.model.component_types.parameter import GenericParameter


class CapacityAdditionUnbounded(GenericParameter):
    """Unbounded capacity addition that can be added each year."""

    name = "capacity_addition_unbounded"
    symbol = r"k^{\mathrm{add,free}}_{h}"
    indices = ("set_technologies",)
    doc = (
        "Parameter which specifies the unbounded capacity addition that can "
        "be added each year (only for delayed technology deployment)."
    )
    unit_category = {"energy_quantity": 1, "time": -1}
