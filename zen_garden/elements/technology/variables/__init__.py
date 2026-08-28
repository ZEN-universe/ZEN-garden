"""Technology variables."""

from zen_garden.model.component_types.variable import GenericVariable

from .capacity import Capacity
from .capacity_addition import CapacityAddition
from .capacity_investment import CapacityInvestment
from .capacity_on_off_helper_var import CapacityOnOffHelperVar
from .capacity_previous import CapacityPrevious
from .carbon_emissions_technology import CarbonEmissionsTechnology
from .carbon_emissions_technology_total import CarbonEmissionsTechnologyTotal
from .cost_capex_overnight import CostCapexOvernight
from .cost_capex_yearly import CostCapexYearly
from .cost_capex_yearly_total import CostCapexYearlyTotal
from .cost_opex_variable import CostOpexVariable
from .cost_opex_yearly import CostOpexYearly
from .cost_opex_yearly_total import CostOpexYearlyTotal
from .tech_on_var import TechOnVar
from .technology_installation import TechnologyInstallation

TECHNOLOGY_VARIABLES: list[type[GenericVariable]] = [
    Capacity,
    CapacityPrevious,
    CapacityAddition,
    CapacityInvestment,
    CostCapexOvernight,
    CostCapexYearly,
    CostCapexYearlyTotal,
    CostOpexVariable,
    CostOpexYearlyTotal,
    CostOpexYearly,
    CarbonEmissionsTechnology,
    CarbonEmissionsTechnologyTotal,
    TechnologyInstallation,
    TechOnVar,
    CapacityOnOffHelperVar,
]

__all__ = [
    "Capacity",
    "CapacityPrevious",
    "CapacityAddition",
    "CapacityInvestment",
    "CostCapexOvernight",
    "CostCapexYearly",
    "CostCapexYearlyTotal",
    "CostOpexVariable",
    "CostOpexYearlyTotal",
    "CostOpexYearly",
    "CarbonEmissionsTechnology",
    "CarbonEmissionsTechnologyTotal",
    "TechnologyInstallation",
    "TechOnVar",
    "CapacityOnOffHelperVar",
    "TECHNOLOGY_VARIABLES",
]
