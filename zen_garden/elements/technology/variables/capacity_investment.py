import numpy as np

from zen_garden.elements.technology.masks import InvestmentEligibleMask
from zen_garden.model.component_types.variable import GenericVariable


class CapacityInvestment(GenericVariable):
    """Variable for invested technology capacity."""

    name = "capacity_investment"
    symbol = r"\Delta K^{\mathrm{inv}}_{h,p,y}"
    indices = ("set_technologies", "set_location", "set_years")
    doc = "Size of invested technology h at location p and period y."
    unit_category = {"energy_quantity": 1, "time": -1}
    bounds = (0, np.inf)
    masks = (InvestmentEligibleMask,)
    mask_default_value = 0.0
