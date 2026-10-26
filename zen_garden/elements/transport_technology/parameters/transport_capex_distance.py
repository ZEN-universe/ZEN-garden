from zen_garden.elements.transport_technology.parameters.capex_per_distance_transport import (  # noqa: E501
    CapexPerDistanceTransport,
)
from zen_garden.elements.transport_technology.parameters.distance import Distance
from zen_garden.model.component_types.parameter import (
    GenericParameter,
    ParameterInputData,
)


class TransportCapexDistance(GenericParameter):
    """Distance-dependent capex contribution of a transport technology.

    Product of the transport distance and the specific capex per distance;
    computed during preprocessing rather than loaded from input data.
    """

    name = "transport_capex_distance"
    symbol = r"k^{\mathrm{distance}}_{h,e,y}"
    indices = ("set_transport_technologies", "set_edges", "set_years")
    doc = "Distance-dependent capital cost for installing a transport technology."
    unit_category = {"money": 1}
    dependencies = (Distance, CapexPerDistanceTransport)

    @classmethod
    def get_input_data(cls, element):
        """Compute distance * capex_per_distance_transport for this element."""
        value = element.capex_per_distance_transport * element.distance
        return ParameterInputData(value, None)
