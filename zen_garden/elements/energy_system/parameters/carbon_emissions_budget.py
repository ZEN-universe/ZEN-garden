from zen_garden.model.component_types.parameter import GenericParameter


class CarbonEmissionsBudget(GenericParameter):
    """Carbon emissions budget over the entire horizon."""

    name = "carbon_emissions_budget"
    symbol = r"\overline{m}^{\mathrm{budget}}"
    indices = ()
    doc = (
        "Parameter which specifies the total budget of carbon emissions until "
        "the end of the entire time horizon."
    )
    unit_category = {"emissions": 1}
