"""Energy system expressions."""

from zen_garden.model.component_types.expression import GenericExpression

from .total_carbon_emissions import TotalCarbonEmissions
from .total_cost import TotalCost

ENERGY_SYSTEM_EXPRESSIONS: list[type[GenericExpression]] = [
    TotalCost,
    TotalCarbonEmissions,
]

__all__ = [
    "TotalCarbonEmissions",
    "TotalCost",
    "ENERGY_SYSTEM_EXPRESSIONS",
]
