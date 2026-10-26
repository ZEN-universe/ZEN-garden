from zen_garden.model.component_types.parameter import GenericParameter


class MinFullLoadHoursFraction(GenericParameter):
    """Minimum full load hours as a fraction of total hours."""

    name = "min_full_load_hours_fraction"
    symbol = r"\ell^{\mathrm{FLH,min}}_{h,n,y}"
    indices = ("set_conversion_technologies", "set_nodes", "set_years")
    doc = (
        "Minimum number of full load hours, expressed as a fraction of the "
        "unaggregated time steps per year."
    )
    unit_category = {}
