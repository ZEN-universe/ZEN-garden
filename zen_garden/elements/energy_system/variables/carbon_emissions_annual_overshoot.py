import numpy as np
from zen_garden.topology.generic_variable import GenericVariable


class CarbonEmissionsAnnualOvershoot(GenericVariable):
    """Variable for annual carbon emissions overshoot."""

    name = "carbon_emissions_annual_overshoot"
    indices = ["set_years"]
    doc = "Variable for overshoot of the annual carbon emissions limit of energy system"
    unit_category = {"emissions": 1}

    @classmethod
    def get_bounds(cls):
        return 0, np.inf
