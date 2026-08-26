from zen_garden.topology.generic_parameter import GenericParameter


class OpexSpecificFixed(GenericParameter):
    """Fixed annual specific opex."""

    name = "opex_specific_fixed"
    indices = ("set_technologies", "set_capacity_types", "set_location", "set_years")
    doc = "Fixed annual specific opex"
    unit_category = {"money": 1, "energy_quantity": -1, "time": 1}
    capacity_types = True
    input_loader = "fixed_opex"
