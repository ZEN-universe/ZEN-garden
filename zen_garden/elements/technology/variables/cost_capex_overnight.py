import numpy as np

from zen_garden.topology.generic_variable import GenericVariable


class CostCapexOvernight(GenericVariable):
    """Variable for capex of building capacity overnight."""

    name = "cost_capex_overnight"
    indices = ["set_technologies", "set_capacity_types", "set_location", "set_years"]
    doc = "Variable for capex for building technology at location l and time t"
    unit_category = {"money": 1}

    @classmethod
    def get_bounds(cls):
        return 0, np.inf
