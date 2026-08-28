from zen_garden.topology.generic_set import GenericSet


class SetEdges(GenericSet):
    name, doc = "set_edges", "Set of edges"

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.network_topology.set_edges
