from zen_garden.model.component_types.variable import GenericVariable


class CarbonEmissionsTechnologyTotal(GenericVariable):
    """Variable for total carbon emissions from technology."""

    name = "carbon_emissions_technology_total"
    symbol = r"M^{\mathrm{tech}}_{y}"
    indices = ["set_years"]
    doc = "Total carbon emissions for operating technology at location p and period y."
    unit_category = {"emissions": 1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
