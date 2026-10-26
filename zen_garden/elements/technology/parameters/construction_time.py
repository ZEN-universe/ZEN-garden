from zen_garden.model.component_types.parameter import GenericParameter


class ConstructionTime(GenericParameter):
    """Construction time of a newly built technology."""

    name = "construction_time"
    symbol = r"L^{\mathrm{con}}_{h}"
    indices = ("set_technologies",)
    doc = "Parameter which specifies the construction time of a newly built technology."
    unit_category = {}
