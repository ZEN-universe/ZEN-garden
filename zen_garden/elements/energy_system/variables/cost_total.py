from zen_garden.model.component_types.variable import GenericVariable


class CostTotal(GenericVariable):
    """Variable for total cost."""

    name = "cost_total"
    symbol = r"C^{\mathrm{total}}_{y}"
    indices = ["set_years"]
    doc = "Total cost of energy system."
    unit_category = {"money": 1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
