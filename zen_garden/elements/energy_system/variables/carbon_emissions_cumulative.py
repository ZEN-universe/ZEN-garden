from zen_garden.model.component_types.variable import GenericVariable


class CarbonEmissionsCumulative(GenericVariable):
    """Variable for cumulative carbon emissions."""

    name = "carbon_emissions_cumulative"
    symbol = r"M^{\mathrm{cum}}_{y}"
    indices = ["set_years"]
    doc = "Cumulative carbon emissions of energy system over time for each year."
    unit_category = {"emissions": 1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
