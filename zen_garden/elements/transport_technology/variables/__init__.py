"""Transport technology variables."""

from zen_garden.topology.generic_variable import GenericVariable

from .flow_transport import FlowTransport
from .flow_transport_loss import FlowTransportLoss

TRANSPORT_TECHNOLOGY_VARIABLES: list[type[GenericVariable]] = [
    FlowTransport,
    FlowTransportLoss,
]

__all__ = [
    "FlowTransport",
    "FlowTransportLoss",
    "TRANSPORT_TECHNOLOGY_VARIABLES",
]

