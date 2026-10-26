from zen_garden.model.component_types.set import GenericSet


class SetTechnologies(GenericSet):
    name = "set_technologies"
    symbol = r"h \in \mathcal{H}"
    doc = "Set of technologies."

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_technologies
