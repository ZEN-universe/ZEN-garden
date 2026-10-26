from zen_garden.model.component_types.variable import GenericVariable


class CostCarrier(GenericVariable):
    """Variable for carrier import/export cost."""

    name = "cost_carrier"
    symbol = r"C^{\mathrm{carrier}}_{c,n,t}"
    indices = [
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    ]
    doc = "Node- and time-dependent carrier cost due to import and export."
    unit_category = {"money": 1, "time": -1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
