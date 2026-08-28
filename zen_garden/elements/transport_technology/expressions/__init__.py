"""Transport technology expressions."""

from zen_garden.model.component_types.expression import GenericExpression

TRANSPORT_TECHNOLOGY_EXPRESSIONS: list[type[GenericExpression]] = []

__all__ = [
    "TRANSPORT_TECHNOLOGY_EXPRESSIONS",
]
