from zen_garden.topology.generic_parameter import GenericParameter


class TimeStepsOperationDuration(GenericParameter):
    """Duration of each operational time step."""

    name = "time_steps_operation_duration"
    indices = ("set_time_steps_operation",)
    doc = "Duration of each operational time step"
    unit_category = {"time": 1}
    set_time_steps = "set_time_steps_operation"
