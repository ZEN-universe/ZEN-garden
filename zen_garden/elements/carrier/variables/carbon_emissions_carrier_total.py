from zen_garden.topology.generic_variable import GenericVariable


class CarbonEmissionsCarrierTotal(GenericVariable):
    """Variable for total carbon emissions from carrier import/export."""

    name = "carbon_emissions_carrier_total"
    indices = ["set_years"]
    doc = "Variable for total carbon emissions of importing and exporting carrier"
    unit_category = {"emissions": 1}

    @classmethod
    def get_bounds(cls):
        return None
