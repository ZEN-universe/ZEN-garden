from zen_garden.topology.generic_variable import GenericVariable


class TechOnVar(GenericVariable):
    """Variable for technology on/off binary."""

    name = "tech_on_var"
    indices = ["set_technologies", "set_location", "set_time_steps_operation"]
    doc = "Binary variable indicating when technology is switched on at location l and time t"
    unit_category = {}
    binary = True

    @classmethod
    def get_bounds(cls):
        return None
