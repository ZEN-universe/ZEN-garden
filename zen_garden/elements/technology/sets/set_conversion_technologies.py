from zen_garden.topology.generic_set import GenericSet


class SetConversionTechnologies(GenericSet):
    name, doc = "set_conversion_technologies", "Set of conversion technologies"

    @classmethod
    def get_data(cls, constructor):
        return constructor.model_schema.set_conversion_technologies
