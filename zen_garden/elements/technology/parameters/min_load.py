from zen_garden.model.component_types.parameter import GenericParameter


class MinLoad(GenericParameter):
    """Minimum load relative to installed capacity."""

    name = "min_load"
    indices = ("set_technologies", "set_location", "set_hours")
    doc = "Minimum load relative to installed capacity"
    unit_category = {}
    time_series = True
