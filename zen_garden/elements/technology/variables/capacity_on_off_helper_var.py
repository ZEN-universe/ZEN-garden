import numpy as np

from zen_garden.elements.technology.variables.tech_on_var import TechOnVar


class CapacityOnOffHelperVar(TechOnVar):
    """Variable for capacity on/off helper."""

    name = "capacity_on_off_helper_var"
    indices = ["set_technologies", "set_location", "set_time_steps_operation"]
    doc = "Variable substituting the product of capacity and tech_on_var"
    unit_category = {"energy_quantity": 1, "time": -1}
    binary = False

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return 0.0, np.inf
