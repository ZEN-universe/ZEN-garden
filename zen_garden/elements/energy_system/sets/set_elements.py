from zen_garden.topology.generic_set import GenericSet


class SetElements(GenericSet):
    name, doc, indexing_set = "set_elements", "Set of elements", True

    @classmethod
    def get_data(cls, constructor):
        return list(
            set(constructor.energy_system.set_technologies)
            | set(constructor.energy_system.set_carriers)
        )
