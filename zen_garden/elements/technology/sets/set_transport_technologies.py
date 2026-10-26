from zen_garden.model.component_types.set import GenericSet


class SetTransportTechnologies(GenericSet):
    name = "set_transport_technologies"
    symbol = r"h \in \mathcal{H}^{\mathrm{trans}}"
    doc = "Set of transport technologies."

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_transport_technologies
