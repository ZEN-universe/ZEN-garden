from zen_garden.model.component_types.parameter import GenericParameter


class CapacityLowerLimitEnergy(GenericParameter):
    """Lower limit on installed storage energy capacity."""

    name = "capacity_lower_limit_energy"
    symbol = r"\underline{e}_{h,n,y}"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "Lower limit on installed storage energy capacity."
    unit_category = {"energy_quantity": 1}
    default_value = 0
    default_unit = "capacity_limit_energy"
