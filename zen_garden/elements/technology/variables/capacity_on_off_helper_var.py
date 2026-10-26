import numpy as np

from zen_garden.elements.technology.variables.tech_on_var import TechOnVar


class CapacityOnOffHelperVar(TechOnVar):
    """Variable for capacity on/off helper."""

    name = "capacity_on_off_helper_var"
    symbol = r"\widehat{K}_{h,p,t}"
    indices = [
        "set_technologies",
        "set_location",
        "set_years",
        "set_representative_hours",
    ]
    doc = "Product of installed capacity and the technology on/off state."
    unit_category = {"energy_quantity": 1, "time": -1}
    variable_type = "continuous"

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return 0.0, np.inf
