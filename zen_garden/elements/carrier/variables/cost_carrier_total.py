from zen_garden.topology.generic_variable import GenericVariable


class CostCarrierTotal(GenericVariable):
    """Variable for total carrier import/export cost."""

    name = "cost_carrier_total"
    indices = ["set_years"]
    doc = "Variable for total carrier cost due to import and export"
    unit_category = {"money": 1}

    @classmethod
    def get_bounds(cls):
        return None

