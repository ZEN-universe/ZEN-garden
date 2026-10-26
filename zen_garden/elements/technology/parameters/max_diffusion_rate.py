from zen_garden.model.component_types.parameter import GenericParameter


class MaxDiffusionRate(GenericParameter):
    """Maximum increase in capacity between investment steps."""

    name = "max_diffusion_rate"
    symbol = r"r^{\mathrm{diff}}_{h}"
    indices = ("set_technologies", "set_years")
    doc = (
        "Parameter which specifies the maximum diffusion rate which is the "
        "maximum increase in capacity between investment steps."
    )
    unit_category = {}
