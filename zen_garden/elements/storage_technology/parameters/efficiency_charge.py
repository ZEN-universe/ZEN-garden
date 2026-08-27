from zen_garden.topology.generic_parameter import GenericParameter


class EfficiencyCharge(GenericParameter):
    """Efficiency during charging."""

    name = "efficiency_charge"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "Efficiency during charging"
    unit_category = {}
