"""conversion technology parameters."""

from zen_garden.model.component_types.parameter import GenericParameter

from .capex_specific_conversion import CapexSpecificConversion
from .conversion_factor import ConversionFactor
from .min_full_load_hours_fraction import MinFullLoadHoursFraction

CONVERSION_TECHNOLOGY_PARAMETERS: list[type[GenericParameter]] = [
    CapexSpecificConversion,
    ConversionFactor,
    MinFullLoadHoursFraction,
]

__all__ = [
    "CapexSpecificConversion",
    "ConversionFactor",
    "MinFullLoadHoursFraction",
    "CONVERSION_TECHNOLOGY_PARAMETERS",
]
