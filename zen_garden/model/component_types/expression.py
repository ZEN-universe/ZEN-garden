"""Generic linear-expression class for OptimizationModel."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, final

import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.model.component_types.component import GenericComponent
from zen_garden.model.linear_expression import (
    transpose_linear_expression,
    validate_component_indices,
)

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor


class GenericExpression(GenericComponent, ABC):
    """Base class for reusable linear expressions.

    An expression is a named piece of a constraint or objective -- a linopy
    ``LinearExpression`` (or ``xarray.DataArray``) built from variables and
    parameters during model construction, after the variables and before the
    constraints. :meth:`build` registers it on the optimization model via
    ``optimization_model.add_expression``; it is then reused elsewhere through
    ``optimization_model.expressions[<name>]``. Concrete expressions document
    their public indices (excluding Linopy's internal ``_term`` dimension) and
    named dependencies as immutable class metadata.
    """

    name: ClassVar[str]
    doc: ClassVar[str]
    indices: ClassVar[tuple[str, ...]]
    dependencies: ClassVar[tuple[type["GenericExpression"], ...]] = ()
    _abstract_component = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("_abstract_component", False):
            return
        if "indices" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define 'indices'")
        if not isinstance(cls.indices, tuple):
            raise TypeError(f"{cls.__name__}.indices must be a tuple")
        if not all(isinstance(index, str) and index for index in cls.indices):
            raise TypeError(f"{cls.__name__}.indices must contain non-empty set names")
        if not isinstance(cls.dependencies, tuple):
            raise TypeError(f"{cls.__name__}.dependencies must be a tuple")
        if "build" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override GenericExpression.build()"
            )

    @classmethod
    @final
    def build(cls, model_constructor: "ModelConstructor") -> None:
        """Compute the expression and register it on the optimization model."""
        if not cls._indices_are_nonempty(
            model_constructor, cls.indices
        ) or not cls.should_construct(model_constructor):
            return
        expression = cls.get_expression(model_constructor)
        canonical_coordinates = {
            index: [
                member
                for member in model_constructor.optimization_model.sets[index]
                if member in expression.indexes[index]
            ]
            for index in cls.indices
            if index in expression.dims
            and index in model_constructor.optimization_model.sets
        }
        expression = expression.sel(canonical_coordinates)
        if isinstance(expression, LinearExpression):
            expression = transpose_linear_expression(expression, cls.indices)
        elif isinstance(expression, xr.DataArray):
            expression = expression.transpose(*cls.indices)
        validate_component_indices(cls.name, cls.indices, expression)
        model_constructor.optimization_model.add_expression(cls, expression)

    @classmethod
    @abstractmethod
    def get_expression(
        cls, model_constructor: "ModelConstructor"
    ) -> LinearExpression | xr.DataArray:
        """Return the linear expression built from variables and parameters."""

    @classmethod
    def should_construct(cls, model_constructor: "ModelConstructor") -> bool:
        """Return whether this expression is needed by the current configuration."""
        return True
