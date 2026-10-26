from zen_garden.model.component_types.parameter import GenericParameter


class CarbonEmissionsAnnualLimit(GenericParameter):
    """Annual carbon emissions limit."""

    name = "carbon_emissions_annual_limit"
    symbol = r"\overline{m}_{y}"
    indices = ("set_years",)
    doc = "Parameter which specifies the total limit on carbon emissions."
    unit_category = {"emissions": 1}
    set_time_steps = "set_years"
