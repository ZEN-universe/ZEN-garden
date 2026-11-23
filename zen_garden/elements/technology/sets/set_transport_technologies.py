from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.model.component_types.set import GenericDomainSet


class SetTransportTechnologies(GenericDomainSet):
    name = "set_transport_technologies"
    symbol = r"h \in \mathcal{H}^{\mathrm{trans}}"
    doc = "Set of transport technologies."
    subset_of = SetTechnologies

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_transport_technologies
