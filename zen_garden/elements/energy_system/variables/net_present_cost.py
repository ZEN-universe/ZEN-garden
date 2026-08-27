from zen_garden.topology.generic_variable import GenericVariable


class NetPresentCost(GenericVariable):
    """Variable for net present cost."""

    name = "net_present_cost"
    indices = ["set_years"]
    doc = "Variable for net_present_cost of energy system"
    unit_category = {"money": 1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
