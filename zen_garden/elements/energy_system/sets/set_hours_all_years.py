from zen_garden.topology.generic_set import GenericSet


class SetHoursAllYears(GenericSet):
    name, doc = "set_hours_all_years", "Set of base time steps"

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_hours_all_years
