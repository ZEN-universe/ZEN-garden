from zen_garden.model.component_types.set import GenericSet


class SetNodesOnEdges(GenericSet):
    name, doc, index_set = (
        "set_nodes_on_edges",
        "Set of nodes that constitute an edge",
        "set_edges",
    )

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.network_topology.set_nodes_on_edges
