"""transport technology parameters."""

from zen_garden.topology.generic_parameter import GenericParameter

from .capex_per_distance_transport import CapexPerDistanceTransport
from .capex_specific_transport import CapexSpecificTransport
from .distance import Distance
from .transport_loss_factor import TransportLossFactor

TRANSPORT_TECHNOLOGY_PARAMETERS: list[type[GenericParameter]] = [
    Distance,
    CapexSpecificTransport,
    CapexPerDistanceTransport,
    TransportLossFactor,
]

__all__ = [
    "Distance",
    "CapexSpecificTransport",
    "CapexPerDistanceTransport",
    "TransportLossFactor",
    "TRANSPORT_TECHNOLOGY_PARAMETERS",
]
