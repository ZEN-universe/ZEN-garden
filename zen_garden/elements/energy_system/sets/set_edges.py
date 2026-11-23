from zen_garden.model.component_types.set import GenericDomainSet

from .set_location import SetLocation


class SetEdges(GenericDomainSet):
    name = "set_edges"
    symbol = r"e \in \mathcal{E}"
    doc = "Set of edges."
    subset_of = SetLocation

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.network_topology.set_edges
