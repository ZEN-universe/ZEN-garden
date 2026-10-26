import numpy as np

from zen_garden.elements.storage_technology.masks.energy_capex_active_mask import (
    EnergyCapexActiveMask,
)
from zen_garden.model.component_types.variable import GenericVariable


class CostCapexYearlyEnergy(GenericVariable):
    """Annualized cost of storage energy capacity."""

    name = "cost_capex_yearly_energy"
    symbol = r"C^{\mathrm{cap,ann,E}}_{h,n,y}"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "Annualized capital cost of storage energy capacity in year y."
    unit_category = {"money": 1}
    bounds = (0, np.inf)
    masks = (EnergyCapexActiveMask,)
    mask_default_value = 0.0
