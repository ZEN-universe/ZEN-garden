"""Conversion technology constraints."""

from zen_garden.model.component_types.constraint import GenericConstraint

from .capacity_factor_conversion_constraint import (
    CapacityFactorConversionConstraint,
)
from .carbon_emissions_technology_conversion_constraint import (
    CarbonEmissionsTechnologyConversionConstraint,
)
from .carrier_conversion_constraint import (
    CarrierConversionConstraint,
)
from .linear_capex_constraint import (
    LinearCapexConstraint,
)
from .minimum_full_load_hours_constraint import (
    MinimumFullLoadHoursConstraint,
)
from .opex_technology_conversion_constraint import OpexTechnologyConversionConstraint

CONVERSION_TECHNOLOGY_CONSTRAINTS: list[type[GenericConstraint]] = [
    CapacityFactorConversionConstraint,
    OpexTechnologyConversionConstraint,
    CarbonEmissionsTechnologyConversionConstraint,
    CarrierConversionConstraint,
    MinimumFullLoadHoursConstraint,
    # LinearCapexConstraint reads only variables/parameters (capacity_addition,
    # cost_capex_overnight, capex_specific_conversion), never state produced by
    # the constraints above, so its position in this list is not load-bearing.
    LinearCapexConstraint,
]

__all__ = [
    "CapacityFactorConversionConstraint",
    "CarbonEmissionsTechnologyConversionConstraint",
    "CarrierConversionConstraint",
    "LinearCapexConstraint",
    "MinimumFullLoadHoursConstraint",
    "OpexTechnologyConversionConstraint",
]
