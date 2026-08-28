from zen_garden.model.component_types.set import GenericSet


class SetYearsEntireHorizon(GenericSet):
    name, doc = (
        "set_years_entire_horizon",
        "Set of yearly time steps of the entire optimization horizon",
    )

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_years_entire_horizon
