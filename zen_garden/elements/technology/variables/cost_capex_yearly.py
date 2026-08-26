import numpy as np

from zen_garden.topology.generic_variable import GenericVariable


class CostCapexYearly(GenericVariable):
    """Variable for annual capex of having capacity."""

    name = "cost_capex_yearly"
    indices = ["set_technologies", "set_capacity_types", "set_location", "set_years"]
    doc = "Variable for annual capex for having technology at location l"
    unit_category = {"money": 1}

    @classmethod
    def get_bounds(cls):
        return 0, np.inf