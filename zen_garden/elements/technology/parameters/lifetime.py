from zen_garden.topology.generic_parameter import GenericParameter


class Lifetime(GenericParameter):
    """Lifetime of a newly built technology."""

    name = "lifetime"
    indices = ("set_technologies",)
    doc = "Lifetime of a newly built technology"
    unit_category = {}
