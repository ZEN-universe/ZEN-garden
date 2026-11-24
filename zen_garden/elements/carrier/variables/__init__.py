"""Carrier variables."""

from zen_garden.model.component_types.variable import GenericVariable

from .flow_export import FlowExport
from .flow_import import FlowImport
from .shed_demand import ShedDemand

CARRIER_VARIABLES: list[type[GenericVariable]] = [
    FlowImport,
    FlowExport,
    ShedDemand,
]

__all__ = [
    "FlowImport",
    "FlowExport",
    "ShedDemand",
    "CARRIER_VARIABLES",
]
