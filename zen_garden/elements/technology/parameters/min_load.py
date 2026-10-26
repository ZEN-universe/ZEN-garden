from zen_garden.model.component_types.parameter import GenericParameter


class MinLoad(GenericParameter):
    """Minimum load relative to installed capacity."""

    name = "min_load"
    symbol = r"\ell^{\mathrm{min}}_{h,p,t}"
    indices = (
        "set_technologies",
        "set_location",
        "set_years",
        "set_representative_hours",
    )
    doc = (
        "Parameter which specifies the minimum load of technology relative to "
        "installed capacity."
    )
    unit_category = {}
    time_series = True
