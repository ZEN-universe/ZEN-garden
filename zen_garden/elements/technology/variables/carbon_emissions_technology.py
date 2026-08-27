from zen_garden.topology.generic_variable import GenericVariable


class CarbonEmissionsTechnology(GenericVariable):
    """Variable for carbon emissions from technology operation."""

    name = "carbon_emissions_technology"
    indices = ["set_technologies", "set_location", "set_time_steps_operation"]
    doc = (
        "Variable for carbon emissions for operating technology at location l and "
        "time t"
    )
    unit_category = {"emissions": 1, "time": -1}

    @classmethod
    def get_bounds(cls):
        return None
