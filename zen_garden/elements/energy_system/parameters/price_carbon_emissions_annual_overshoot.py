from zen_garden.topology.generic_parameter import GenericParameter


class PriceCarbonEmissionsAnnualOvershoot(GenericParameter):
    """Carbon price for annual overshoot."""

    name = "price_carbon_emissions_annual_overshoot"
    indices = ()
    doc = "Carbon price for annual overshoot"
    unit_category = {"money": 1, "emissions": -1}
