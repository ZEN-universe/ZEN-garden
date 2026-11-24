"""Generic linear-expression class for OptimizationModel."""

from abc import ABC, abstractmethod
from inspect import cleandoc
from typing import TYPE_CHECKING, ClassVar, final

import xarray as xr
from linopy.expressions import LinearExpression
from linopy.variables import Variable

from zen_garden.model.component_types.component import GenericComponent
from zen_garden.model.component_types.mask import GenericMask
from zen_garden.model.component_types.set import GenericRelationSet
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
    unit_category: ClassVar[dict[str, int]]
    dependencies: ClassVar[tuple[type["GenericExpression"], ...]] = ()
    masks: ClassVar[tuple[type[GenericMask], ...]] = ()
    index_relations: ClassVar[str | tuple[type[GenericRelationSet], ...]] = "auto"
    mask_default_value: ClassVar[float | None] = None
    _abstract_component = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("_abstract_component", False):
            return
        if "doc" not in cls.__dict__ and cls.get_expression.__doc__:
            cls.doc = cleandoc(cls.get_expression.__doc__)
        for attribute in ("indices", "unit_category"):
            if attribute not in cls.__dict__:
                raise TypeError(f"{cls.__name__} must define {attribute!r}")
        if "symbol" in cls.__dict__:
            raise TypeError(f"{cls.__name__} must not define 'symbol'")
        if not isinstance(cls.indices, tuple):
            raise TypeError(f"{cls.__name__}.indices must be a tuple")
        if not all(isinstance(index, str) and index for index in cls.indices):
            raise TypeError(f"{cls.__name__}.indices must contain non-empty set names")
        if not isinstance(cls.unit_category, dict) or not all(
            isinstance(dimension, str) and isinstance(exponent, int)
            for dimension, exponent in cls.unit_category.items()
        ):
            raise TypeError(
                f"{cls.__name__}.unit_category must map dimensions to integers"
            )
        if not isinstance(cls.dependencies, tuple):
            raise TypeError(f"{cls.__name__}.dependencies must be a tuple")
        if "build" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override GenericExpression.build()"
            )
        cls.masks = tuple(cls.__dict__.get("masks", ()))
        if cls.masks and cls.mask_default_value is None:
            raise TypeError(
                f"{cls.__name__} must declare mask_default_value when using masks"
            )

    @classmethod
    @final
    def build(cls, model_constructor: "ModelConstructor") -> None:
        """Compute the expression and register it on the optimization model."""
        model_constructor.optimization_model.expressions.declare(cls)
        if not cls._indices_are_nonempty(
            model_constructor, cls.indices
        ) or not cls.should_construct(model_constructor):
            return
        expression = cls.get_expression(model_constructor)
        if (
            isinstance(expression, LinearExpression)
            and "_term" in expression.const.dims
        ):
            expression = LinearExpression(
                xr.Dataset(
                    {
                        "coeffs": expression.coeffs,
                        "vars": expression.vars,
                        "const": expression.const.sum("_term"),
                    }
                ),
                expression.model,
            )
        resolved = None
        performance_mask = None
        if cls.masks:
            resolved = model_constructor.optimization_model.sets.resolve_indices(
                cls.indices, index_relations=cls.index_relations
            )
            performance_mask = (
                model_constructor.optimization_model.masks.combine_for_indices(
                    cls.masks, resolved.structural_mask
                )
            )
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
        model_constructor.optimization_model.add_expression(
            cls,
            expression,
            structural_mask=(
                resolved.structural_mask if resolved is not None else None
            ),
            performance_mask=performance_mask,
            mask_default_value=cls.mask_default_value,
        )

    @classmethod
    @abstractmethod
    def get_expression(
        cls, model_constructor: "ModelConstructor"
    ) -> LinearExpression | Variable | xr.DataArray:
        """Return the linear expression built from variables and parameters."""

    @classmethod
    def should_construct(cls, model_constructor: "ModelConstructor") -> bool:
        """Return whether this expression is needed by the current configuration."""
        return True
