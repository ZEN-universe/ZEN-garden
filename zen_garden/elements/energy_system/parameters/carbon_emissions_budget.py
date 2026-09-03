from zen_garden.model.component_types.parameter import GenericParameter


class CarbonEmissionsBudget(GenericParameter):
    """Carbon emissions budget over the entire horizon."""

    name = "carbon_emissions_budget"
    indices = ()
    doc = "Carbon emissions budget over the entire horizon"
    unit_category = {"emissions": 1}
