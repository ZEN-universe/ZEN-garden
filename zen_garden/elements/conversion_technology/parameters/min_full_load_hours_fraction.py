from zen_garden.model.component_types.parameter import GenericParameter


class MinFullLoadHoursFraction(GenericParameter):
    """Minimum full load hours as a fraction of total hours."""

    name = "min_full_load_hours_fraction"
    indices = ("set_conversion_technologies", "set_nodes", "set_years")
    doc = "Minimum full load hours as a fraction of total hours"
    unit_category = {}
