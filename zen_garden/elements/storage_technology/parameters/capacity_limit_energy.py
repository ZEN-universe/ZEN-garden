from zen_garden.model.component_types.parameter import GenericParameter


class CapacityLimitEnergy(GenericParameter):
    """Upper limit on installed storage energy capacity."""

    name = "capacity_limit_energy"
    symbol = r"\overline{e}_{h,n,y}"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "Upper limit on installed storage energy capacity."
    unit_category = {"energy_quantity": 1}
