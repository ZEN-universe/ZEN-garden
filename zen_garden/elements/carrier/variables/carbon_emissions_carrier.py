from zen_garden.model.component_types.variable import GenericVariable


class CarbonEmissionsCarrier(GenericVariable):
    """Variable for carbon emissions from carrier import/export."""

    name = "carbon_emissions_carrier"
    indices = ["set_carriers", "set_nodes", "set_time_steps_operation"]
    doc = "Variable for carbon emissions of importing and exporting carrier"
    unit_category = {"emissions": 1, "time": -1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
