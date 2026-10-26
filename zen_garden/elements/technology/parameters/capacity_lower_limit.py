from zen_garden.model.component_types.parameter import GenericParameter


class CapacityLowerLimit(GenericParameter):
    """Lower capacity limit of technologies."""

    name = "capacity_lower_limit"
    symbol = r"\underline{k}_{h,p,y}"
    indices = ("set_technologies", "set_location", "set_years")
    doc = "Parameter which specifies the capacity lower limit of technologies."
    unit_category = {"energy_quantity": 1, "time": -1}
    default_value = 0
    default_unit = "capacity_limit"
