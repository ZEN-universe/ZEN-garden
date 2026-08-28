import numpy as np

from zen_garden.model.component_types.variable import GenericVariable


class CapacityPrevious(GenericVariable):
    """Variable for installed technology capacity from previous year."""

    name = "capacity_previous"
    indices = ["set_technologies", "set_capacity_types", "set_location", "set_years"]
    doc = "Variable for size of installed technology at location l and BEFORE time t"
    unit_category = {"energy_quantity": 1, "time": -1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return 0, np.inf
