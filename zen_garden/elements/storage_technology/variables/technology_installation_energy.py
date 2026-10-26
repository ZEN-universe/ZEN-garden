from zen_garden.elements.storage_technology.masks.energy_capacity_addition_eligible_mask import (  # noqa: E501
    EnergyCapacityAdditionEligibleMask,
)
from zen_garden.elements.storage_technology.masks.energy_installation_relevant_mask import (  # noqa: E501
    EnergyInstallationRelevantMask,
)
from zen_garden.model.component_types.variable import GenericVariable


class TechnologyInstallationEnergy(GenericVariable):
    """Whether storage energy capacity is installed in a year."""

    name = "technology_installation_energy"
    symbol = r"z^{\mathrm{install,E}}_{h,n,y}"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "One when storage energy capacity is installed at node n in year y."
    unit_category = {}
    variable_type = "binary"
    masks = (
        EnergyCapacityAdditionEligibleMask,
        EnergyInstallationRelevantMask,
    )
    mask_default_value = 0.0
