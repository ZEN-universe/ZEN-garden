"""Conversion technology expressions."""

from zen_garden.model.component_types.expression import GenericExpression

CONVERSION_TECHNOLOGY_EXPRESSIONS: list[type[GenericExpression]] = []

__all__ = [
    "CONVERSION_TECHNOLOGY_EXPRESSIONS",
]
