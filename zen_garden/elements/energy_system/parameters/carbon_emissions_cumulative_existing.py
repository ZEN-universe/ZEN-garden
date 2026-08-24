from zen_garden.topology.generic_parameter import GenericParameter


class CarbonEmissionsCumulativeExisting(GenericParameter):
    """Previous cumulative carbon emissions."""

    name = "carbon_emissions_cumulative_existing"
    indices = ()
    doc = "Previous cumulative carbon emissions"
    unit_category = {"emissions": 1}
