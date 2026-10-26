from zen_garden.model.component_types.parameter import GenericParameter


class PriceCarbonEmissions(GenericParameter):
    """Yearly carbon price."""

    name = "price_carbon_emissions"
    symbol = r"\pi^{\mathrm{CO_2}}_{y}"
    indices = ("set_years",)
    doc = "Parameter which specifies the yearly carbon price."
    unit_category = {"money": 1, "emissions": -1}
    set_time_steps = "set_years"
