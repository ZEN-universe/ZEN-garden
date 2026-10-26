from zen_garden.model.component_types.variable import GenericVariable


class CarbonEmissionsCarrier(GenericVariable):
    """Variable for carbon emissions from carrier import/export."""

    name = "carbon_emissions_carrier"
    symbol = r"M^{\mathrm{carrier}}_{c,n,t}"
    indices = [
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    ]
    doc = "Carbon emissions of importing and exporting carriers."
    unit_category = {"emissions": 1, "time": -1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
