from zen_garden.model.component_types.parameter import GenericParameter


class DiscountRate(GenericParameter):
    """Discount rate of the energy system."""

    name = "discount_rate"
    symbol = r"r^{\mathrm{disc}}"
    indices = ()
    doc = "Parameter which specifies the discount rate of the energy system."
    unit_category = {}
