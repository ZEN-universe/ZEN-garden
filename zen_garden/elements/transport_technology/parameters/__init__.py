"""transport technology parameters."""

from zen_garden.model.component_types.parameter import GenericParameter

from .capex_per_distance_transport import CapexPerDistanceTransport
from .capex_specific_transport import CapexSpecificTransport
from .distance import Distance
from .transport_capex_distance import TransportCapexDistance
from .transport_loss_factor import TransportLossFactor

TRANSPORT_TECHNOLOGY_PARAMETERS: list[type[GenericParameter]] = [
    Distance,
    CapexSpecificTransport,
    CapexPerDistanceTransport,
    TransportLossFactor,
    TransportCapexDistance,
]

__all__ = [
    "Distance",
    "CapexSpecificTransport",
    "CapexPerDistanceTransport",
    "TransportLossFactor",
    "TransportCapexDistance",
    "TRANSPORT_TECHNOLOGY_PARAMETERS",
]
