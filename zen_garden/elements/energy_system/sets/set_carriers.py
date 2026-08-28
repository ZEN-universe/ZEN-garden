from zen_garden.topology.generic_set import GenericSet


class SetCarriers(GenericSet):
    name, doc = "set_carriers", "Set of carriers"

    @classmethod
    def get_data(cls, constructor):
        return constructor.model_schema.set_carriers
