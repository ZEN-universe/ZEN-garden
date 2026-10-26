import numpy as np

from zen_garden.elements.storage_technology.masks.energy_capacity_addition_eligible_mask import (  # noqa: E501
    EnergyCapacityAdditionEligibleMask,
)
from zen_garden.model.component_types.variable import GenericVariable


class CostCapexOvernightEnergy(GenericVariable):
    """Overnight cost of storage energy-capacity investments."""

    name = "cost_capex_overnight_energy"
    symbol = r"C^{\mathrm{cap,overnight,E}}_{h,n,y}"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "Overnight cost of storage energy capacity completed in year y."
    unit_category = {"money": 1}
    bounds = (0, np.inf)
    masks = (EnergyCapacityAdditionEligibleMask,)
    mask_default_value = 0.0
