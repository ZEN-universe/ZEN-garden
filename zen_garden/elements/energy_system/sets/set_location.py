from zen_garden.model.component_types.set import GenericDomainSet


class SetLocation(GenericDomainSet):
    """Canonical domain containing every node and directed edge."""

    name = "set_location"
    symbol = r"p \in \mathcal{P}"
    doc = "Set of model locations, comprising nodes and directed edges."

    @classmethod
    def get_data(cls, model_constructor):
        topology = model_constructor.network_topology
        return [*topology.set_nodes, *topology.set_edges]
