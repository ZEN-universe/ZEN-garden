from zen_garden.model.component_types.variable import GenericVariable


class CostCarbonEmissionsTotal(GenericVariable):
    """Variable for total cost of carbon emissions."""

    name = "cost_carbon_emissions_total"
    symbol = r"C^{\mathrm{CO_2}}_{y}"
    indices = ["set_years"]
    doc = "Total cost of carbon emissions of energy system."
    unit_category = {"money": 1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
