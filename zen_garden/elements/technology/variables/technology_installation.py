from zen_garden.elements.technology.masks import (
    CapacityAdditionEligibleMask,
    InstallationRelevantMask,
)
from zen_garden.model.component_types.variable import GenericVariable


class TechnologyInstallation(GenericVariable):
    """Variable for technology installation."""

    name = "technology_installation"
    indices = ["set_technologies", "set_capacity_types", "set_location", "set_years"]
    doc = "Binary variable indicating installation of technology"
    unit_category = {}
    binary = True
    masks = [CapacityAdditionEligibleMask, InstallationRelevantMask]

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
