"""Registry for reusable expressions constructed by the optimization model."""

from collections.abc import Iterator, Mapping

import xarray as xr
from linopy.expressions import LinearExpression
from linopy.variables import Variable

from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.registries.base import component_name

ExpressionValue = LinearExpression | Variable | xr.DataArray
ExpressionKey = str | type[GenericExpression]


class ExpressionRegistry(Mapping[str, ExpressionValue]):
    """Store expressions by name while accepting expression classes as keys."""

    def __init__(self) -> None:
        self._expressions: dict[str, ExpressionValue] = {}
        self.specifications: dict[str, type[GenericExpression]] = {}
        self.docs: dict[str, str] = {}
        self.indices: dict[str, tuple[str, ...]] = {}
        self.unit_categories: dict[str, dict[str, int]] = {}
        self.structural_masks: dict[str, xr.DataArray] = {}
        self.performance_masks: dict[str, xr.DataArray] = {}
        self.mask_default_values: dict[str, float] = {}

    def declare(self, specification: type[GenericExpression]) -> None:
        """Register expression metadata even when its index sets are empty."""
        name = specification.name
        existing = self.specifications.get(name)
        if existing is not None and existing is not specification:
            raise ValueError(f"Expression {name!r} is declared more than once")
        self.specifications[name] = specification
        self.docs[name] = specification.doc
        self.indices[name] = specification.indices
        self.unit_categories[name] = specification.unit_category

    def add(
        self,
        key: ExpressionKey,
        expression: ExpressionValue,
        *,
        structural_mask: xr.DataArray | None = None,
        performance_mask: xr.DataArray | None = None,
        mask_default_value: float | None = None,
    ) -> None:
        """Register an expression under its canonical name."""
        name = component_name(key, GenericExpression)
        if name in self._expressions:
            raise ValueError(f"Expression {name!r} already added")
        self._expressions[name] = expression
        if isinstance(key, type):
            self.declare(key)
        if structural_mask is not None:
            self.structural_masks[name] = structural_mask
        if performance_mask is not None:
            self.performance_masks[name] = performance_mask
        if mask_default_value is not None:
            self.mask_default_values[name] = mask_default_value

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

    def select(self, requested: list[str]) -> list[str]:
        """Validate a configured selection and return constructible names.

        An empty selection means every expression constructed for the current
        model. A declared expression whose index sets are empty is valid but is
        omitted because it has no value to evaluate.
        """
        unknown = sorted(set(requested).difference(self.specifications))
        if unknown:
            raise ValueError(
                f"Unknown selected_saved_expressions: {unknown}. "
                f"Available expressions: {sorted(self.specifications)}"
            )
        if not requested:
            return list(self)
        return [name for name in requested if name in self]
