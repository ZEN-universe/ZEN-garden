"""technology parameters."""

from zen_garden.topology.generic_parameter import GenericParameter

from .capacity_addition_max import CapacityAdditionMax
from .capacity_addition_min import CapacityAdditionMin
from .capacity_addition_unbounded import CapacityAdditionUnbounded
from .capacity_existing import CapacityExisting
from .capacity_investment_existing import CapacityInvestmentExisting
from .capacity_limit import CapacityLimit
from .capacity_lower_limit import CapacityLowerLimit
from .capex_capacity_existing import CapexCapacityExisting
from .carbon_intensity_technology import CarbonIntensityTechnology
from .construction_time import ConstructionTime
from .depreciation_time import DepreciationTime
from .lifetime import Lifetime
from .lifetime_existing import LifetimeExisting
from .max_diffusion_rate import MaxDiffusionRate
from .max_load import MaxLoad
from .min_load import MinLoad
from .opex_specific_fixed import OpexSpecificFixed
from .opex_specific_variable import OpexSpecificVariable

TECHNOLOGY_PARAMETERS: list[type[GenericParameter]] = [
    CapacityExisting,
    CapacityInvestmentExisting,
    CapacityAdditionMin,
    CapacityAdditionMax,
    CapacityAdditionUnbounded,
    LifetimeExisting,
    CapexCapacityExisting,
    OpexSpecificVariable,
    OpexSpecificFixed,
    Lifetime,
    DepreciationTime,
    ConstructionTime,
    MaxDiffusionRate,
    CapacityLimit,
    CapacityLowerLimit,
    MinLoad,
    MaxLoad,
    CarbonIntensityTechnology,
]

__all__ = [
    "CapacityExisting",
    "CapacityInvestmentExisting",
    "CapacityAdditionMin",
    "CapacityAdditionMax",
    "CapacityAdditionUnbounded",
    "LifetimeExisting",
    "CapexCapacityExisting",
    "OpexSpecificVariable",
    "OpexSpecificFixed",
    "Lifetime",
    "DepreciationTime",
    "ConstructionTime",
    "MaxDiffusionRate",
    "CapacityLimit",
    "CapacityLowerLimit",
    "MinLoad",
    "MaxLoad",
    "CarbonIntensityTechnology",
    "TECHNOLOGY_PARAMETERS",
]
