"""Registry for reusable expressions constructed by the optimization model."""

from collections.abc import Iterator, Mapping

import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.registries.base import component_name

ExpressionValue = LinearExpression | xr.DataArray
ExpressionKey = str | type[GenericExpression]


class ExpressionRegistry(Mapping[str, ExpressionValue]):
    """Store expressions by name while accepting expression classes as keys."""

    def __init__(self) -> None:
        self._expressions: dict[str, ExpressionValue] = {}

    def add(self, key: ExpressionKey, expression: ExpressionValue) -> None:
        """Register an expression under its canonical name."""
        name = component_name(key, GenericExpression)
        if name in self._expressions:
            raise ValueError(f"Expression {name!r} already added")
        self._expressions[name] = expression

    def __getitem__(self, key: ExpressionKey) -> ExpressionValue:
        """Return an expression by name or declarative expression class."""
        return self._expressions[component_name(key, GenericExpression)]

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return key in self._expressions
        if isinstance(key, type) and issubclass(key, GenericExpression):
            return key.name in self._expressions
        return False

    def __iter__(self) -> Iterator[str]:
        return iter(self._expressions)

    def __len__(self) -> int:
        return len(self._expressions)
