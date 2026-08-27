from zen_garden.topology.generic_set import GenericSet


class SetRetrofittingTechnologies(GenericSet):
    name, doc = "set_retrofitting_technologies", "Set of retrofitting technologies"

    @classmethod
    def get_data(cls, constructor):
        return constructor.energy_system.set_retrofitting_technologies
