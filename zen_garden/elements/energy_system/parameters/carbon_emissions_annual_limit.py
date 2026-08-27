from zen_garden.topology.generic_parameter import GenericParameter


class CarbonEmissionsAnnualLimit(GenericParameter):
    """Annual carbon emissions limit."""

    name = "carbon_emissions_annual_limit"
    indices = ("set_years",)
    doc = "Annual carbon emissions limit"
    unit_category = {"emissions": 1}
    set_time_steps = "set_years"
