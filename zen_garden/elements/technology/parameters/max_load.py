from zen_garden.model.component_types.parameter import GenericParameter


class MaxLoad(GenericParameter):
    """Maximum load relative to installed capacity."""

    name = "max_load"
    symbol = r"\ell^{\mathrm{max}}_{h,p,t}"
    indices = (
        "set_technologies",
        "set_location",
        "set_years",
        "set_representative_hours",
    )
    doc = (
        "Parameter which specifies the maximum load of technology relative to "
        "installed capacity."
    )
    unit_category = {}
    time_series = True
