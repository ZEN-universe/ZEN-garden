import numpy as np

from zen_garden.elements.technology.masks import CapacityAnyTypeMask
from zen_garden.model.component_types.variable import GenericVariable


class CostOpexYearly(GenericVariable):
    """Variable for yearly operational expenditure."""

    name = "cost_opex_yearly"
    symbol = r"C^{\mathrm{op,ann}}_{h,p,y}"
    indices = ["set_technologies", "set_location", "set_years"]
    doc = "Yearly opex for operating technology h at location p and year y."
    unit_category = {"money": 1}
    masks = [CapacityAnyTypeMask]
    mask_default_value = 0.0

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return 0, np.inf
