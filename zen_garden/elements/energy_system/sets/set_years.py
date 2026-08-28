from zen_garden.model.component_types.set import GenericSet


class SetYears(GenericSet):
    name, doc = "set_years", "Set of yearly time steps"

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_years
