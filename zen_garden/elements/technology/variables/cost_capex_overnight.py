import numpy as np

from zen_garden.elements.technology.masks import CapacityAdditionEligibleMask
from zen_garden.model.component_types.variable import GenericVariable


class CostCapexOvernight(GenericVariable):
    """Variable for capex of building capacity overnight."""

    name = "cost_capex_overnight"
    symbol = r"C^{\mathrm{cap,overnight}}_{h,p,y}"
    indices = ("set_technologies", "set_location", "set_years")
    doc = "Capex for building technology h at location p and period y."
    unit_category = {"money": 1}
    bounds = (0, np.inf)
    masks = (CapacityAdditionEligibleMask,)
    mask_default_value = 0.0
