import numpy as np

from zen_garden.topology.generic_variable import GenericVariable


class CostOpexYearlyTotal(GenericVariable):
    """Variable for total operational expenditure."""

    name = "cost_opex_yearly_total"
    indices = ["set_years"]
    doc = "Variable for total opex all technologies and locations in year y"
    unit_category = {"money": 1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return 0, np.inf
