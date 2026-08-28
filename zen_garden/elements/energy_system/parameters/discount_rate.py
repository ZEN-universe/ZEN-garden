from zen_garden.model.component_types.parameter import GenericParameter


class DiscountRate(GenericParameter):
    """Discount rate of the energy system."""

    name = "discount_rate"
    indices = ()
    doc = "Discount rate of the energy system"
    unit_category = {}
