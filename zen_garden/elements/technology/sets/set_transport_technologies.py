from zen_garden.topology.generic_set import GenericSet


class SetTransportTechnologies(GenericSet):
    name, doc = "set_transport_technologies", "Set of transport technologies"

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_transport_technologies
