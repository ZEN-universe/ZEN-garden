from zen_garden.model.component_types.parameter import GenericParameter


class CarbonEmissionsCumulativeExisting(GenericParameter):
    """Previous cumulative carbon emissions."""

    name = "carbon_emissions_cumulative_existing"
    symbol = r"m^{\mathrm{cum}}_{0}"
    indices = ()
    doc = "Parameter which specifies the total previous carbon emissions."
    unit_category = {"emissions": 1}
