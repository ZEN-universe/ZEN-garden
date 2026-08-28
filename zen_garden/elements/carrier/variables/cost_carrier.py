from zen_garden.model.component_types.variable import GenericVariable


class CostCarrier(GenericVariable):
    """Variable for carrier import/export cost."""

    name = "cost_carrier"
    indices = ["set_carriers", "set_nodes", "set_time_steps_operation"]
    doc = "Variable for node- and time-dependent carrier cost due to import and export"
    unit_category = {"money": 1, "time": -1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
