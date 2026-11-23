from zen_garden.model.component_types.set import GenericDomainSet

from .set_location import SetLocation


class SetNodes(GenericDomainSet):
    name = "set_nodes"
    symbol = r"n \in \mathcal{N}"
    doc = "Set of nodes."
    subset_of = SetLocation

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.network_topology.set_nodes
