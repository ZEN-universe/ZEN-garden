from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.model.component_types.variable import GenericVariable


class CarbonEmissionsTechnology(GenericVariable):
    """Variable for carbon emissions from technology operation."""

    name = "carbon_emissions_technology"
    symbol = r"M^{\mathrm{tech}}_{h,p,t}"
    indices = [
        "set_technologies",
        "set_location",
        "set_years",
        "set_representative_hours",
    ]
    doc = "Carbon emissions for operating technology h at location p and time t."
    unit_category = {"emissions": 1, "time": -1}
    masks = [PowerOperationMask]
    mask_default_value = 0.0

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None
