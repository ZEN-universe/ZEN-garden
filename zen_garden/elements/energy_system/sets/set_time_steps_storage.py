from zen_garden.topology.generic_set import GenericSet


class SetTimeStepsStorage(GenericSet):
    name, doc = "set_time_steps_storage", "Set of storage level time steps"

    @classmethod
    def get_data(cls, constructor):
        return constructor.model_schema.time_steps.time_steps_storage
