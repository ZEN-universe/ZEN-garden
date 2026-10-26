import numpy as np

from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.model.component_types.variable import GenericVariable


class CostOpexVariable(GenericVariable):
    """Variable for operational expenditure."""

    name = "cost_opex_variable"
    symbol = r"C^{\mathrm{op,var}}_{h,p,t}"
    indices = [
        "set_technologies",
        "set_location",
        "set_years",
        "set_representative_hours",
    ]
    doc = "Opex for operating technology h at location p and operational time step t."
    unit_category = {"money": 1, "time": -1}
    masks = [PowerOperationMask]
    mask_default_value = 0.0

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return 0, np.inf
