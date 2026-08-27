import numpy as np

from zen_garden.topology.generic_variable import GenericVariable


class CostOpexYearly(GenericVariable):
    """Variable for yearly operational expenditure."""

    name = "cost_opex_yearly"
    indices = ["set_technologies", "set_location", "set_years"]
    doc = "Variable for yearly opex for operating technology at location l and year y"
    unit_category = {"money": 1}

    @classmethod
    def get_bounds(cls):
        return 0, np.inf
