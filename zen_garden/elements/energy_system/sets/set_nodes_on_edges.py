from zen_garden.elements.energy_system.sets.set_edges import SetEdges
from zen_garden.model.component_types.set import GenericSet


class SetNodesOnEdges(GenericSet):
    name = "set_nodes_on_edges"
    symbol = r"n \in \mathcal{N}_{e}"
    doc = "Set of nodes that constitute an edge."
    index_set = SetEdges

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.network_topology.set_nodes_on_edges
