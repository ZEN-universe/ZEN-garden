from zen_garden.topology.generic_parameter import GenericParameter


class MaxDiffusionRate(GenericParameter):
    """Maximum increase in capacity between investment steps."""

    name = "max_diffusion_rate"
    indices = ("set_technologies", "set_years")
    doc = "Maximum increase in capacity between investment steps"
    unit_category = {}
