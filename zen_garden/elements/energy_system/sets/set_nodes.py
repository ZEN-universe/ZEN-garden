from zen_garden.model.component_types.set import GenericSet


class SetNodes(GenericSet):
    name = "set_nodes"
    symbol = r"n \in \mathcal{N}"
    doc = "Set of nodes."

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.network_topology.set_nodes
