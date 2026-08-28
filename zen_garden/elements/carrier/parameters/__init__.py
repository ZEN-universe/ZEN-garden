"""Carrier parameters."""

from zen_garden.model.component_types.parameter import GenericParameter

from .availability_export import AvailabilityExport
from .availability_export_yearly import AvailabilityExportYearly
from .availability_import import AvailabilityImport
from .availability_import_yearly import AvailabilityImportYearly
from .carbon_intensity_carrier_export import CarbonIntensityCarrierExport
from .carbon_intensity_carrier_import import CarbonIntensityCarrierImport
from .demand import Demand
from .price_export import PriceExport
from .price_import import PriceImport
from .price_shed_demand import PriceShedDemand

CARRIER_PARAMETERS: list[type[GenericParameter]] = [
    AvailabilityExport,
    AvailabilityExportYearly,
    AvailabilityImport,
    AvailabilityImportYearly,
    CarbonIntensityCarrierExport,
    CarbonIntensityCarrierImport,
    Demand,
    PriceExport,
    PriceImport,
    PriceShedDemand,
]

__all__ = [
    "AvailabilityExport",
    "AvailabilityExportYearly",
    "AvailabilityImport",
    "AvailabilityImportYearly",
    "CarbonIntensityCarrierExport",
    "CarbonIntensityCarrierImport",
    "Demand",
    "PriceExport",
    "PriceImport",
    "PriceShedDemand",
    "CARRIER_PARAMETERS",
]
