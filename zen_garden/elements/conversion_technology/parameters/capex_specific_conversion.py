from zen_garden.model.component_types.parameter import GenericParameter


class CapexSpecificConversion(GenericParameter):
    """Slope of the linear capex."""

    name = "capex_specific_conversion"
    indices = ("set_conversion_technologies", "set_nodes", "set_years")
    doc = "Slope of the linear capex"
    unit_category = {"money": 1, "energy_quantity": -1, "time": 1}
