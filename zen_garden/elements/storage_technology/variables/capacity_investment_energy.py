import numpy as np

from zen_garden.elements.storage_technology.masks.energy_investment_eligible_mask import (  # noqa: E501
    EnergyInvestmentEligibleMask,
)
from zen_garden.model.component_types.variable import GenericVariable


class CapacityInvestmentEnergy(GenericVariable):
    """Investment in storage energy capacity."""

    name = "capacity_investment_energy"
    symbol = r"\Delta E^{\mathrm{inv}}_{h,n,y}"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "Investment in storage energy capacity at node n in year y."
    unit_category = {"energy_quantity": 1}
    bounds = (0, np.inf)
    masks = (EnergyInvestmentEligibleMask,)
    mask_default_value = 0.0
