"""Storage technology expressions."""

from zen_garden.topology.generic_expression import GenericExpression

STORAGE_TECHNOLOGY_EXPRESSIONS: list[type[GenericExpression]] = []

__all__ = [
    "STORAGE_TECHNOLOGY_EXPRESSIONS",
]
