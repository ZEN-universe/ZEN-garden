from zen_garden.topology.generic_set import GenericSet


class SetNodes(GenericSet):
    name, doc = "set_nodes", "Set of nodes"

    @classmethod
    def get_data(cls, constructor):
        return constructor.energy_system.set_nodes
