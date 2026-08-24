from zen_garden.topology.generic_parameter import GenericParameter


class TimeStepsStorageDuration(GenericParameter):
    """Duration of each storage time step."""

    name = "time_steps_storage_duration"
    indices = ("set_time_steps_storage",)
    doc = "Duration of each storage time step"
    unit_category = {"time": 1}
    set_time_steps = "set_time_steps_storage"
