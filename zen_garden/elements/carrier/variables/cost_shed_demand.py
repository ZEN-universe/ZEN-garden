import numpy as np

from zen_garden.model.component_types.variable import GenericVariable


class CostShedDemand(GenericVariable):
    """Variable for cost of shedding demand of carrier."""

    name = "cost_shed_demand"
    symbol = r"C^{\mathrm{shed}}_{c,n,t}"
    indices = [
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    ]
    doc = "Cost shed carrier demand."
    unit_category = {"money": 1, "time": -1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return 0.0, np.inf
