from zen_garden.topology.generic_set import GenericSet


class SetNodesOnEdges(GenericSet):
    name, doc, index_set = (
        "set_nodes_on_edges",
        "Set of nodes that constitute an edge",
        "set_edges",
    )

    @classmethod
    def get_data(cls, constructor):
        return constructor.network_topology.set_nodes_on_edges
