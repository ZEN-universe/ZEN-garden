import numpy as np
from zen_garden.topology.generic_variable import GenericVariable


class CostCapexYearlyTotal(GenericVariable):
    """Variable for total capex."""

    name = "cost_capex_yearly_total"
    indices = ["set_years"]
    doc = "Variable for total capex for installing all technologies in all locations at all times"
    unit_category = {"money": 1}

    @classmethod
    def get_bounds(cls):
        return 0, np.inf
