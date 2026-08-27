from zen_garden.topology.generic_parameter import GenericParameter


class PriceCarbonEmissions(GenericParameter):
    """Yearly carbon price."""

    name = "price_carbon_emissions"
    indices = ("set_years",)
    doc = "Yearly carbon price"
    unit_category = {"money": 1, "emissions": -1}
    set_time_steps = "set_years"
