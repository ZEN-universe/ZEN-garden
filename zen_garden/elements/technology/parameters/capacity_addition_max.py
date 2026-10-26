from zen_garden.model.component_types.parameter import GenericParameter


class CapacityAdditionMax(GenericParameter):
    """Parameter which specifies the maximum capacity addition that can be installed."""

    name = "capacity_addition_max"
    symbol = r"\overline{\Delta k}_{h,p,y}"
    indices = ("set_technologies",)
    doc = (
        "Parameter which specifies the maximum capacity addition that can be installed."
    )
    unit_category = {"energy_quantity": 1, "time": -1}
