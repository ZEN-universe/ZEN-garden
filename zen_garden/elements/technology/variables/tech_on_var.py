from zen_garden.topology.generic_variable import GenericVariable


class TechOnVar(GenericVariable):
    """Variable for technology on/off binary."""

    name = "tech_on_var"
    indices = ["set_technologies", "set_location", "set_time_steps_operation"]
    doc = "Variable for binary indicator when technology is switched on at location l and time t"
    unit_category = None
    binary = True

    @classmethod
    def get_bounds(cls):
        return None

