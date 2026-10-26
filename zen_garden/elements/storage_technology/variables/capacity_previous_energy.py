import numpy as np

from zen_garden.elements.storage_technology.masks.energy_capacity_available_mask import (  # noqa: E501
    EnergyCapacityAvailableMask,
)
from zen_garden.model.component_types.variable import GenericVariable


class CapacityPreviousEnergy(GenericVariable):
    """Storage energy capacity available before additions in a year."""

    name = "capacity_previous_energy"
    symbol = r"E^{\mathrm{prev}}_{h,n,y}"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "Storage energy capacity available before additions in year y."
    unit_category = {"energy_quantity": 1}
    bounds = (0, np.inf)
    masks = (EnergyCapacityAvailableMask,)
    mask_default_value = 0.0
