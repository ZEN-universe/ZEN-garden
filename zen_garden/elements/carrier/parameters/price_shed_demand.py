from zen_garden.model.component_types.parameter import GenericParameter


class PriceShedDemand(GenericParameter):
    """Parameter which specifies the price of shed demand."""

    name = "price_shed_demand"
    indices = ("set_carriers",)
    doc = "Parameter which specifies the price of shed demand"
    unit_category = {"money": 1, "energy_quantity": -1}
