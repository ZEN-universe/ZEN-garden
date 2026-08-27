from zen_garden.topology.generic_parameter import GenericParameter


class ConstructionTime(GenericParameter):
    """Construction time of a newly built technology."""

    name = "construction_time"
    indices = ("set_technologies",)
    doc = "Construction time of a newly built technology"
    unit_category = {}
