import numpy as np

from zen_garden.topology.generic_variable import GenericVariable


class CarbonEmissionsBudgetOvershoot(GenericVariable):
    """Variable for carbon emissions budget overshoot."""

    name = "carbon_emissions_budget_overshoot"
    indices = ["set_years"]
    doc = (
        "Variable for overshoot carbon emissions of energy system at the end of the "
        "time horizon"
    )
    unit_category = {"emissions": 1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return 0, np.inf
