"""Transport technology variables."""

from zen_garden.model.component_types.variable import GenericVariable

from .flow_transport import FlowTransport

TRANSPORT_TECHNOLOGY_VARIABLES: list[type[GenericVariable]] = [
    FlowTransport,
]

__all__ = [
    "FlowTransport",
    "TRANSPORT_TECHNOLOGY_VARIABLES",
]
