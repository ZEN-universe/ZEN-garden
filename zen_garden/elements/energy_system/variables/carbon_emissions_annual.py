from zen_garden.topology.generic_variable import GenericVariable


class CarbonEmissionsAnnual(GenericVariable):
    """Variable for annual carbon emissions."""

    name = "carbon_emissions_annual"
    indices = ["set_years"]
    doc = "Variable for annual carbon emissions of energy system"
    unit_category = {"emissions": 1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
