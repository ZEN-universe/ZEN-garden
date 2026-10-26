from zen_garden.model.component_types.set import GenericSet


class SetRetrofittingTechnologies(GenericSet):
    name = "set_retrofitting_technologies"
    symbol = r"h \in \mathcal{H}^{\mathrm{retro}}"
    doc = "Set of retrofitting technologies."

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_retrofitting_technologies
