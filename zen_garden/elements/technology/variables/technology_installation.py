from zen_garden.topology.generic_variable import GenericVariable


class TechnologyInstallation(GenericVariable):
    """Variable for technology installation binary."""

    name = "technology_installation"
    indices = ["set_technologies", "set_capacity_types", "set_location", "set_years"]
    doc = "Variable for installment of a technology at location l and time t"
    unit_category = None
    binary = True

    @classmethod
    def get_bounds(cls):
        return None

