from zen_garden.model.component_types.parameter import GenericParameter


class CapexSpecificConversion(GenericParameter):
    """Slope of the linear capex."""

    name = "capex_specific_conversion"
    symbol = r"\kappa^{\mathrm{cap}}_{h,y}"
    indices = ("set_conversion_technologies", "set_nodes", "set_years")
    doc = "Parameter which specifies the slope of the capex if approximated linearly."
    unit_category = {"money": 1, "energy_quantity": -1, "time": 1}
