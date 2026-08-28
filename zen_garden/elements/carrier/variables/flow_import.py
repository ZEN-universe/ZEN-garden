import numpy as np

from zen_garden.model.component_types.variable import GenericVariable


class FlowImport(GenericVariable):
    """Variable for import flow."""

    name = "flow_import"
    indices = ["set_carriers", "set_nodes", "set_time_steps_operation"]
    doc = "Variable for node- and time-dependent carrier import from the grid"
    unit_category = {"energy_quantity": 1, "time": -1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return 0.0, np.inf
