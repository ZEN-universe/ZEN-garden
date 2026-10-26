import numpy as np

from zen_garden.elements.technology.masks import CapacityAvailableMask
from zen_garden.model.component_types.variable import GenericVariable


class CapacityPrevious(GenericVariable):
    """Variable for installed technology capacity from previous year."""

    name = "capacity_previous"
    symbol = r"K^{\mathrm{prev}}_{h,p,y}"
    indices = ("set_technologies", "set_location", "set_years")
    doc = "Size of installed technology h at location p before additions in period y."
    unit_category = {"energy_quantity": 1, "time": -1}
    bounds = (0, np.inf)
    masks = (CapacityAvailableMask,)
    mask_default_value = 0.0
