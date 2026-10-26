import numpy as np

from zen_garden.elements.technology.masks import CapexActiveMask
from zen_garden.model.component_types.variable import GenericVariable


class CostCapexYearly(GenericVariable):
    """Variable for annual capex of having capacity."""

    name = "cost_capex_yearly"
    symbol = r"C^{\mathrm{cap,ann}}_{h,p,y}"
    indices = ("set_technologies", "set_location", "set_years")
    doc = "Annual capex for having technology h at location p."
    unit_category = {"money": 1}
    bounds = (0, np.inf)
    masks = (CapexActiveMask,)
    mask_default_value = 0.0
