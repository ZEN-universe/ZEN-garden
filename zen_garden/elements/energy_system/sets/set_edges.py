from zen_garden.topology.generic_set import GenericSet


class SetEdges(GenericSet):
    name, doc = "set_edges", "Set of edges"

    @classmethod
    def get_data(cls, constructor):
        return constructor.model_schema.set_edges
