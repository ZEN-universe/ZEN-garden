"""Technology expressions."""

from zen_garden.model.component_types.expression import GenericExpression

TECHNOLOGY_EXPRESSIONS: list[type[GenericExpression]] = []

__all__ = [
    "TECHNOLOGY_EXPRESSIONS",
]
