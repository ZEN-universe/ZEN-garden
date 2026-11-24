"""Technology variables."""

from zen_garden.model.component_types.variable import GenericVariable

from .capacity import Capacity
from .capacity_addition import CapacityAddition
from .capacity_investment import CapacityInvestment
from .capacity_on_off_helper_var import CapacityOnOffHelperVar
from .tech_on_var import TechOnVar
from .technology_installation import TechnologyInstallation

TECHNOLOGY_VARIABLES: list[type[GenericVariable]] = [
    Capacity,
    CapacityAddition,
    CapacityInvestment,
    TechnologyInstallation,
    TechOnVar,
    CapacityOnOffHelperVar,
]

__all__ = [
    "Capacity",
    "CapacityAddition",
    "CapacityInvestment",
    "TechnologyInstallation",
    "TechOnVar",
    "CapacityOnOffHelperVar",
    "TECHNOLOGY_VARIABLES",
]
