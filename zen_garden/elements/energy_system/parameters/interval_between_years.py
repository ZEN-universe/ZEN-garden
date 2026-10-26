from zen_garden.model.component_types.parameter import GenericParameter, ParameterData


class IntervalBetweenYears(GenericParameter):
    """Number of calendar years represented by one planning period."""

    name = "interval_between_years"
    symbol = r"\Delta y"
    indices = ()
    doc = "Number of calendar years represented by one planning period."
    unit_category = {"time": 1}

    @classmethod
    def get_parameter_value(cls, model_constructor):
        """Return the configured interval as a scalar parameter."""
        return ParameterData(
            model_constructor.config.system.interval_between_years, cls.indices, {}
        )

    @classmethod
    def get_input_data(cls, element):
        """Return no input because the value comes from configuration."""
        return None
