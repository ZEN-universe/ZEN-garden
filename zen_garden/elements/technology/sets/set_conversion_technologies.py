from zen_garden.model.component_types.set import GenericSet


class SetConversionTechnologies(GenericSet):
    name, doc = "set_conversion_technologies", "Set of conversion technologies"

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_conversion_technologies
