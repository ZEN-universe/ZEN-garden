"""Carrier variables."""

from zen_garden.topology.generic_variable import GenericVariable

from .carbon_emissions_carrier import CarbonEmissionsCarrier
from .carbon_emissions_carrier_total import CarbonEmissionsCarrierTotal
from .cost_carrier import CostCarrier
from .cost_carrier_total import CostCarrierTotal
from .cost_shed_demand import CostShedDemand
from .flow_export import FlowExport
from .flow_import import FlowImport
from .shed_demand import ShedDemand

CARRIER_VARIABLES: list[type[GenericVariable]] = [
    FlowImport,
    FlowExport,
    CostCarrier,
    CostCarrierTotal,
    CarbonEmissionsCarrier,
    CarbonEmissionsCarrierTotal,
    ShedDemand,
    CostShedDemand,
]

__all__ = [
    "FlowImport",
    "FlowExport",
    "CostCarrier",
    "CostCarrierTotal",
    "CarbonEmissionsCarrier",
    "CarbonEmissionsCarrierTotal",
    "ShedDemand",
    "CostShedDemand",
    "CARRIER_VARIABLES",
]
