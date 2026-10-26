from zen_garden.model.component_types.parameter import GenericParameter


class WeightRepresentativeHours(GenericParameter):
    """Amount of original time represented by each operational hour."""

    name = "weight_representative_hours"
    symbol = r"w_r"
    indices = ("set_years", "set_representative_hours")
    doc = "Number of original hours represented by a representative hour."
    unit_category = {"time": 1}
    squeeze_singleton = False

    @classmethod
    def get_input_data(cls, element):
        """Retain weights populated by time-series aggregation."""
        return None
