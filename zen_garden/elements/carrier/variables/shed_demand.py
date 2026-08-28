import numpy as np

from zen_garden.model.component_types.variable import GenericVariable


class ShedDemand(GenericVariable):
    """Variable for shedding carrier demand."""

    name = "shed_demand"
    indices = ["set_carriers", "set_nodes", "set_time_steps_operation"]
    doc = "Variable for shedding demand of carrier"
    unit_category = {"energy_quantity": 1, "time": -1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return 0.0, np.inf
