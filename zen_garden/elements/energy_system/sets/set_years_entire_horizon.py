from zen_garden.model.component_types.set import GenericSet


class SetYearsEntireHorizon(GenericSet):
    name = "set_years_entire_horizon"
    symbol = r"y \in \mathcal{Y}^{\mathrm{horizon}}"
    doc = (
        "All yearly timesteps, regardless of whether they are inside or "
        "outside the current foresight horizon."
    )

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_years_entire_horizon
