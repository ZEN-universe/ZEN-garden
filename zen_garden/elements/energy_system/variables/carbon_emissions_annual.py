from zen_garden.model.component_types.variable import GenericVariable


class CarbonEmissionsAnnual(GenericVariable):
    """Variable for annual carbon emissions."""

    name = "carbon_emissions_annual"
    symbol = r"M_{y}"
    indices = ["set_years"]
    doc = "Annual carbon emissions of energy system."
    unit_category = {"emissions": 1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
