from zen_garden.topology.generic_set import GenericSet


class SetCarriers(GenericSet):
    name, doc = "set_carriers", "Set of carriers"

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_carriers
