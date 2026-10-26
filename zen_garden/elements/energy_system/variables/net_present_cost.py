from zen_garden.model.component_types.variable import GenericVariable


class NetPresentCost(GenericVariable):
    """Variable for net present cost."""

    name = "net_present_cost"
    symbol = r"C^{\mathrm{NPC}}_{y}"
    indices = ["set_years"]
    doc = "Net_present_cost of energy system."
    unit_category = {"money": 1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
