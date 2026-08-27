from zen_garden.topology.generic_set import GenericSet


class SetYears(GenericSet):
    name, doc = "set_years", "Set of yearly time steps"

    @classmethod
    def get_data(cls, constructor):
        return constructor.energy_system.set_years
