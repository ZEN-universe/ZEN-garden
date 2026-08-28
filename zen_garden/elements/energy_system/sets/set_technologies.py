from zen_garden.topology.generic_set import GenericSet


class SetTechnologies(GenericSet):
    name, doc = "set_technologies", "Set of technologies"

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_technologies
