"""Carrier expressions."""

from zen_garden.model.component_types.expression import GenericExpression

CARRIER_EXPRESSIONS: list[type[GenericExpression]] = []

__all__ = [
    "CARRIER_EXPRESSIONS",
]
