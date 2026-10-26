from zen_garden.model.component_types.parameter import GenericParameter, ParameterData


class ModeledYearHours(GenericParameter):
    """Number of operational hours represented by a modeled year."""

    name = "modeled_year_hours"
    symbol = r"T^{\mathrm{model}}"
    indices = ()
    doc = "Number of operational hours represented by a modeled year."
    unit_category = {"time": 1}

    @classmethod
    def get_parameter_value(cls, model_constructor):
        """Return the configured modeled-year duration as a scalar parameter."""
        value = model_constructor.config.system.unaggregated_time_steps_per_year
        return ParameterData(value, cls.indices, {})

    @classmethod
    def get_input_data(cls, element):
        """Return no input because the value comes from configuration."""
        return None
