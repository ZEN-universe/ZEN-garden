from zen_garden.topology.generic_variable import GenericVariable


class TechnologyInstallation(GenericVariable):
    """Variable for technology installation."""

    name = "technology_installation"
    indices = ["set_technologies", "set_capacity_types", "set_location", "set_years"]
    doc = "Binary variable indicating installation of technology"
    unit_category = {}
    binary = True

    @classmethod
    def get_bounds(cls):
        return None

