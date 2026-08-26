from zen_garden.topology.generic_set import GenericSet


class SetYearsEntireHorizon(GenericSet):
    name, doc = (
        "set_years_entire_horizon",
        "Set of yearly time steps of the entire optimization horizon",
    )

    @classmethod
    def get_data(cls, constructor):
        return constructor.energy_system.set_years_entire_horizon
