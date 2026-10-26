from zen_garden.model.component_types.variable import GenericVariable


class CostCarrierTotal(GenericVariable):
    """Variable for total carrier import/export cost."""

    name = "cost_carrier_total"
    symbol = r"C^{\mathrm{carrier}}_{y}"
    indices = ["set_years"]
    doc = "Total carrier cost due to import and export."
    unit_category = {"money": 1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
