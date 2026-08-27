from zen_garden.topology.generic_variable import GenericVariable


class CarbonEmissionsTechnologyTotal(GenericVariable):
    """Variable for total carbon emissions from technology."""

    name = "carbon_emissions_technology_total"
    indices = ["set_years"]
    doc = "Variable for total carbon emissions for operating technology"
    unit_category = {"emissions": 1}

    @classmethod
    def get_bounds(cls):
        return None
