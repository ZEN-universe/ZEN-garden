from zen_garden.topology.generic_set import GenericSet


class SetTimeStepsOperation(GenericSet):
    name, doc = "set_time_steps_operation", "Set of operational time steps"

    @classmethod
    def get_data(cls, constructor):
        return constructor.time_steps.time_steps_operation
