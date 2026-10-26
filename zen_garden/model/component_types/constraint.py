"""Generic constraint class for OptimizationModel."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, final

from linopy.constraints import Constraint

from zen_garden.model.component_types.component import GenericComponent
from zen_garden.model.component_types.mask import GenericMask
from zen_garden.model.linear_expression import (
    transpose_constraint,
    validate_component_indices,
)

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor


class GenericConstraint(GenericComponent, ABC):
    """Base class for constraints.

    Constraints are stateless: :meth:`build` is a final template method that
    conditionally constructs, masks, and registers exactly one relation.
    """

    name: ClassVar[str]
    doc: ClassVar[str]
    indices: ClassVar[tuple[str, ...]]
    masks: ClassVar[tuple[type["GenericMask"], ...]] = ()
    _abstract_component = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("_abstract_component", False):
            return
        if "build" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override GenericConstraint.build()"
            )
        if "indices" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define 'indices'")
        if not isinstance(cls.indices, tuple):
            raise TypeError(f"{cls.__name__}.indices must be a tuple")
        if not all(isinstance(index, str) and index for index in cls.indices):
            raise TypeError(f"{cls.__name__}.indices must contain non-empty set names")
        if not isinstance(cls.masks, tuple):
            raise TypeError(f"{cls.__name__}.masks must be a tuple")
        if not all(
            isinstance(mask, type) and issubclass(mask, GenericMask)
            for mask in cls.masks
        ):
            raise TypeError(f"{cls.__name__}.masks must contain GenericMask classes")

    @classmethod
    @final
    def build(cls, model_constructor: "ModelConstructor") -> None:
        """Construct and register this class's single constraint."""
        if not cls._indices_are_nonempty(
            model_constructor, cls.indices
        ) or not cls.should_construct(model_constructor):
            return

        optimization_model = model_constructor.optimization_model
        constraint = cls.get_constraint(model_constructor)
        constraint = transpose_constraint(constraint, cls.indices)
        validate_component_indices(cls.name, cls.indices, constraint)
        mask = optimization_model.masks.combine_for_constraint(
            list(cls.masks), constraint
        )
        if mask is not None:
            optimization_model.masks.log_reduction(cls.name, mask)
        optimization_model.add_constraint(cls.name, constraint, doc=cls.doc, mask=mask)

    @classmethod
    def should_construct(cls, model_constructor: "ModelConstructor") -> bool:
        """Return whether the constraint applies to the current model."""
        return True

    @classmethod
    @abstractmethod
    def get_constraint(cls, model_constructor: "ModelConstructor") -> Constraint:
        """Return this class's mathematical constraint relation."""
