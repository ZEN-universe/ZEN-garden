"""Conversion Technology constraints."""

from zen_garden.constraints.generic_constraint import GenericConstraint
from zen_garden.elements.conversion_technology.constraints.capacity_factor_conversion_constraint import (
    CapacityFactorConversionConstraint,
)
from zen_garden.elements.conversion_technology.constraints.carrier_conversion_constraint import (
    CarrierConversionConstraint,
)
from zen_garden.elements.conversion_technology.constraints.linear_capex_constraint import (
    LinearCapexConstraint,
)
from zen_garden.elements.conversion_technology.constraints.minimum_full_load_hours_constraint import (
    MinimumFullLoadHoursConstraint,
)
from zen_garden.elements.conversion_technology.constraints.opex_emissions_technology_conversion_constraint import (
    OpexEmissionsTechnologyConversionConstraint,
)
from zen_garden.elements.conversion_technology.conversion_technology import (
    ConversionTechnology,
)

CONVERSION_TECHNOLOGY_CONSTRAINTS: list[type[GenericConstraint]] = [
    CapacityFactorConversionConstraint,
    OpexEmissionsTechnologyConversionConstraint,
    CarrierConversionConstraint,
    MinimumFullLoadHoursConstraint,
]

__all__ = [
    "ConversionTechnology",
    "CapacityCapexCouplingConstraint",
    "CapacityFactorConversionConstraint",
    "CarrierConversionConstraint",
    "LinearCapexConstraint",
    "MinimumFullLoadHoursConstraint",
    "OpexEmissionsTechnologyConversionConstraint",
]
