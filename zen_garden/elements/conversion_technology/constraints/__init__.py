"""Conversion technology constraints."""

from zen_garden.model.component_types.constraint import GenericConstraint

from .capacity_factor_conversion_constraint import (
    CapacityFactorConversionConstraint,
)
from .carrier_conversion_constraint import (
    CarrierConversionConstraint,
)
from .minimum_full_load_hours_constraint import (
    MinimumFullLoadHoursConstraint,
)

CONVERSION_TECHNOLOGY_CONSTRAINTS: list[type[GenericConstraint]] = [
    CapacityFactorConversionConstraint,
    CarrierConversionConstraint,
    MinimumFullLoadHoursConstraint,
]

__all__ = [
    "CapacityFactorConversionConstraint",
    "CarrierConversionConstraint",
    "MinimumFullLoadHoursConstraint",
]
