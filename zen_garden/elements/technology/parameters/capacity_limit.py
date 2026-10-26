from zen_garden.model.component_types.parameter import GenericParameter


class CapacityLimit(GenericParameter):
    """Capacity limit of technologies."""

    name = "capacity_limit"
    symbol = r"\overline{k}_{h,p,y}"
    indices = ("set_technologies", "set_location", "set_years")
    doc = "Parameter which specifies the capacity limit of technologies."
    unit_category = {"energy_quantity": 1, "time": -1}
