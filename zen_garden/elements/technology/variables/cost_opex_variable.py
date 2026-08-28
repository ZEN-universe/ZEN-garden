import numpy as np

from zen_garden.model.component_types.variable import GenericVariable


class CostOpexVariable(GenericVariable):
    """Variable for operational expenditure."""

    name = "cost_opex_variable"
    indices = ["set_technologies", "set_location", "set_time_steps_operation"]
    doc = "Variable for opex for operating technology at location l and time t"
    unit_category = {"money": 1, "time": -1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return 0, np.inf
