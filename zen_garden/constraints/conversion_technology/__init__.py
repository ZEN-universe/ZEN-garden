from zen_garden.constraints.generic_constraint import GenericConstraint

from .capacity_capex_coupling_constraint import CapacityCapexCouplingConstraint
from .capacity_factor_conversion_constraint import CapacityFactorConversionConstraint
from .carrier_conversion_constraint import CarrierConversionConstraint
from .linear_capex_constraint import LinearCapexConstraint
from .minimum_full_load_hours_constraint import MinimumFullLoadHoursConstraint
from .opex_emissions_technology_conversion_constraint import (
    OpexEmissionsTechnologyConversionConstraint,
)

CONVERSION_TECHNOLOGY_CONSTRAINTS: list[type[GenericConstraint]] = [
    CapacityFactorConversionConstraint,
    OpexEmissionsTechnologyConversionConstraint,
    CarrierConversionConstraint,
    MinimumFullLoadHoursConstraint,
]

__all__ = [
    "CapacityCapexCouplingConstraint",
    "CapacityFactorConversionConstraint",
    "CarrierConversionConstraint",
    "LinearCapexConstraint",
    "MinimumFullLoadHoursConstraint",
    "OpexEmissionsTechnologyConversionConstraint",
]
