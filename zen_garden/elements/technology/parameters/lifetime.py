from zen_garden.model.component_types.parameter import GenericParameter


class Lifetime(GenericParameter):
    """Lifetime of a newly built technology."""

    name = "lifetime"
    symbol = r"L_{h}"
    indices = ("set_technologies",)
    doc = "Parameter which specifies the lifetime of a newly built technology."
    unit_category = {}
