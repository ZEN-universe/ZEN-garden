from zen_garden.topology.generic_variable import GenericVariable


class CostCarbonEmissionsTotal(GenericVariable):
    """Variable for total cost of carbon emissions."""

    name = "cost_carbon_emissions_total"
    indices = ["set_years"]
    doc = "Variable for total cost of carbon emissions of energy system"
    unit_category = {"money": 1}

    @classmethod
    def get_bounds(cls):
        return None
