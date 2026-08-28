from zen_garden.model.component_types.parameter import GenericParameter


class ConstructionTime(GenericParameter):
    """Construction time of a newly built technology."""

    name = "construction_time"
    indices = ("set_technologies",)
    doc = "Construction time of a newly built technology"
    unit_category = {}
