from zen_garden.model.component_types.variable import GenericVariable


class CarbonEmissionsCarrierTotal(GenericVariable):
    """Variable for total carbon emissions from carrier import/export."""

    name = "carbon_emissions_carrier_total"
    symbol = r"M^{\mathrm{carrier}}_{y}"
    indices = ["set_years"]
    doc = "Total carbon emissions of importing and exporting carriers."
    unit_category = {"emissions": 1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
