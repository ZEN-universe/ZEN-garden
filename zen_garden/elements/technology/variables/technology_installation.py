from zen_garden.elements.technology.masks import (
    CapacityAdditionEligibleMask,
    InstallationRelevantMask,
)
from zen_garden.model.component_types.variable import GenericVariable


class TechnologyInstallation(GenericVariable):
    """Variable for technology installation."""

    name = "technology_installation"
    symbol = r"z^{\mathrm{install}}_{h,p,y}"
    indices = ("set_technologies", "set_location", "set_years")
    doc = (
        "Binary variable which equals 1 if technology h is installed at "
        "location p and period y, else 0."
    )
    unit_category = {}
    variable_type = "binary"
    masks = (CapacityAdditionEligibleMask, InstallationRelevantMask)
    mask_default_value = 0.0

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
