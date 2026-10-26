"""Carrier expressions."""

from zen_garden.model.component_types.expression import GenericExpression

from .annual_carrier_cost import AnnualCarrierCost
from .annual_export_quantity import AnnualExportQuantity
from .annual_import_export_emissions import AnnualImportExportEmissions
from .annual_import_quantity import AnnualImportQuantity
from .nodal_conversion_consumption import NodalConversionConsumption
from .nodal_conversion_production import NodalConversionProduction
from .nodal_net_supply import NodalNetSupply
from .nodal_storage_charging import NodalStorageCharging
from .nodal_storage_discharging import NodalStorageDischarging
from .nodal_transport_inflow import NodalTransportInflow
from .nodal_transport_outflow import NodalTransportOutflow

CARRIER_EXPRESSIONS: list[type[GenericExpression]] = [
    AnnualImportQuantity,
    AnnualExportQuantity,
    AnnualCarrierCost,
    AnnualImportExportEmissions,
    NodalConversionConsumption,
    NodalConversionProduction,
    NodalTransportInflow,
    NodalTransportOutflow,
    NodalStorageCharging,
    NodalStorageDischarging,
    NodalNetSupply,
]

__all__ = [
    "AnnualCarrierCost",
    "AnnualExportQuantity",
    "AnnualImportExportEmissions",
    "AnnualImportQuantity",
    "CARRIER_EXPRESSIONS",
    "NodalConversionConsumption",
    "NodalConversionProduction",
    "NodalNetSupply",
    "NodalStorageCharging",
    "NodalStorageDischarging",
    "NodalTransportInflow",
    "NodalTransportOutflow",
]
