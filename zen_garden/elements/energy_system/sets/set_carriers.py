from zen_garden.model.component_types.set import GenericSet


class SetCarriers(GenericSet):
    name = "set_carriers"
    symbol = r"c \in \mathcal{C}"
    doc = "Set of carriers."

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_carriers
