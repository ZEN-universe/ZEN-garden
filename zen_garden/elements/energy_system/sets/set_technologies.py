from zen_garden.topology.generic_set import GenericSet


class SetTechnologies(GenericSet):
    name, doc = "set_technologies", "Set of technologies"

    @classmethod
    def get_data(cls, constructor):
        return constructor.energy_system.set_technologies
