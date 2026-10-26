import numpy as np

from zen_garden.model.component_types.variable import GenericVariable


class CarbonEmissionsAnnualOvershoot(GenericVariable):
    """Variable for annual carbon emissions overshoot."""

    name = "carbon_emissions_annual_overshoot"
    symbol = r"M^{\mathrm{ann,over}}_{y}"
    indices = ["set_years"]
    doc = "Overshoot of the annual carbon emissions limit of energy system."
    unit_category = {"emissions": 1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return 0, np.inf
