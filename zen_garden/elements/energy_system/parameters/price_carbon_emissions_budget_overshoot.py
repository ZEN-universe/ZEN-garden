from zen_garden.model.component_types.parameter import GenericParameter


class PriceCarbonEmissionsBudgetOvershoot(GenericParameter):
    """Carbon price for budget overshoot."""

    name = "price_carbon_emissions_budget_overshoot"
    symbol = r"\pi^{\mathrm{CO_2,bud}}"
    indices = ()
    doc = "Parameter which specifies the carbon price for budget overshoot."
    unit_category = {"money": 1, "emissions": -1}
