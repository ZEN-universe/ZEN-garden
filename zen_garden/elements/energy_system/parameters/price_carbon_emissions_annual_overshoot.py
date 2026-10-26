from zen_garden.model.component_types.parameter import GenericParameter


class PriceCarbonEmissionsAnnualOvershoot(GenericParameter):
    """Carbon price for annual overshoot."""

    name = "price_carbon_emissions_annual_overshoot"
    symbol = r"\pi^{\mathrm{CO_2,ann}}"
    indices = ()
    doc = "Parameter which specifies the carbon price for annual overshoot."
    unit_category = {"money": 1, "emissions": -1}
