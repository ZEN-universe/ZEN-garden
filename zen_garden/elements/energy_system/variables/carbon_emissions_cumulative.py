from zen_garden.topology.generic_variable import GenericVariable


class CarbonEmissionsCumulative(GenericVariable):
    """Variable for cumulative carbon emissions."""

    name = "carbon_emissions_cumulative"
    indices = ["set_years"]
    doc = (
        "Variable for cumulative carbon emissions of energy system over time for each "
        "year"
    )
    unit_category = {"emissions": 1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
