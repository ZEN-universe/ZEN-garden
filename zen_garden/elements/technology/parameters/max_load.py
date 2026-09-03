from zen_garden.model.component_types.parameter import GenericParameter


class MaxLoad(GenericParameter):
    """Maximum load relative to installed capacity."""

    name = "max_load"
    indices = ("set_technologies", "set_location", "set_hours")
    doc = "Maximum load relative to installed capacity"
    unit_category = {}
    time_series = True
