from zen_garden.model.component_types.parameter import GenericParameter


class PriceCarbonEmissionsBudgetOvershoot(GenericParameter):
    """Carbon price for budget overshoot."""

    name = "price_carbon_emissions_budget_overshoot"
    indices = ()
    doc = "Carbon price for budget overshoot"
    unit_category = {"money": 1, "emissions": -1}
