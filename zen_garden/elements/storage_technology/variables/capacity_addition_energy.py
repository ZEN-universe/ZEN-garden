import numpy as np

from zen_garden.elements.storage_technology.masks.energy_capacity_addition_eligible_mask import (  # noqa: E501
    EnergyCapacityAdditionEligibleMask,
)
from zen_garden.model.component_types.variable import GenericVariable


class CapacityAdditionEnergy(GenericVariable):
    """Storage energy capacity completed in a modeled year."""

    name = "capacity_addition_energy"
    symbol = r"\Delta E_{h,n,y}"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "Storage energy capacity completed at node n in year y."
    unit_category = {"energy_quantity": 1}
    bounds = (0, np.inf)
    masks = (EnergyCapacityAdditionEligibleMask,)
    mask_default_value = 0.0
