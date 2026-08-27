from zen_garden.topology.generic_variable import GenericVariable


class CostTotal(GenericVariable):
    """Variable for total cost."""

    name = "cost_total"
    indices = ["set_years"]
    doc = "Variable for total cost of energy system"
    unit_category = {"money": 1}

    @classmethod
    def get_bounds(cls):
        return None
