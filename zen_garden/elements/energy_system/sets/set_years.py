from zen_garden.model.component_types.set import GenericSet


class SetYears(GenericSet):
    name = "set_years"
    symbol = r"y \in \mathcal{Y}"
    doc = (
        "Set of yearly timesteps within the current foresight horizon. For "
        "simulations with myopic foresight, this set is a subset "
        ":math:`\\mathcal{Y}^\\mathrm{horizon}`."
    )

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.model_schema.set_years
