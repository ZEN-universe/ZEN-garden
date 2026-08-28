from zen_garden.model.component_types.parameter import GenericParameter


class TransportCapexDistance(GenericParameter):
    """Distance-dependent capex contribution of a transport technology.

    Product of the transport distance and the specific capex per distance;
    computed during preprocessing rather than loaded from input data.
    """

    name = "transport_capex_distance"
    indices = ("set_transport_technologies", "set_edges", "set_years")
    doc = "Transport distance times the specific capex per distance"
    unit_category = {"money": 1}
    dependencies = ["distance", "capex_per_distance_transport"]

    @classmethod
    def store_input_data(cls, element):
        """Compute distance * capex_per_distance_transport for this element."""
        value = element.capex_per_distance_transport * element.distance
        cls._store_value(element, cls.name, value)
