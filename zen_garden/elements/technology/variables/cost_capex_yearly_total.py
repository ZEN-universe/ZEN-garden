import numpy as np

from zen_garden.model.component_types.variable import GenericVariable


class CostCapexYearlyTotal(GenericVariable):
    """Variable for total capex."""

    name = "cost_capex_yearly_total"
    symbol = r"C^{\mathrm{cap}}_{y}"
    indices = ["set_years"]
    doc = (
        "Total capex for installing all technologies in all locations at all "
        "times in period y."
    )
    unit_category = {"money": 1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return 0, np.inf
