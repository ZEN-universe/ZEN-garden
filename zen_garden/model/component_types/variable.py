from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, TypeAlias, final

import numpy as np
import xarray as xr

from zen_garden.model.component_types.component import GenericComponent
from zen_garden.model.component_types.mask import GenericMask
from zen_garden.model.component_types.set import GenericRelationSet

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor


VariableType = Literal["continuous", "binary", "integer"]
VariableIndexSet: TypeAlias = tuple[list[Any], list[str]]
BoundValue: TypeAlias = int | float | np.ndarray | xr.DataArray
BoundFunction: TypeAlias = Callable[..., tuple[int | float, int | float]]
VariableBounds: TypeAlias = (
    tuple[BoundValue, BoundValue] | np.ndarray | BoundFunction | None
)


class GenericVariable(GenericComponent, ABC):
    """Abstract base class for variables in ZEN-garden."""

    name: ClassVar[str]
    symbol: ClassVar[str]
    indices: ClassVar[Sequence[str]]
    doc: ClassVar[str]
    unit_category: ClassVar[dict[str, int]]
    variable_type: ClassVar[VariableType] = "continuous"
    bounds: ClassVar[VariableBounds] = None
    masks: ClassVar[Sequence[type["GenericMask"]]] = ()
    index_relations: ClassVar[str | tuple[type[GenericRelationSet], ...]] = "auto"
    mask_default_value: ClassVar[float | None] = None
    _abstract_component = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls.__dict__.get("_abstract_component", False):
            return
        required = ("symbol", "indices", "unit_category")

        for attr in required:
            if attr not in cls.__dict__:
                raise TypeError(f"{cls.__name__} must define {attr!r}")
        for attr in ("symbol",):
            if (
                not isinstance(getattr(cls, attr), str)
                or not getattr(cls, attr).strip()
            ):
                raise TypeError(f"{cls.__name__}.{attr} must be a non-empty string")
        if isinstance(cls.indices, str) or not all(
            isinstance(index, str) and index for index in cls.indices
        ):
            raise TypeError(f"{cls.__name__}.indices must contain non-empty set names")
        cls.indices = tuple(cls.indices)
        if not isinstance(cls.unit_category, dict) or not all(
            isinstance(dimension, str) and isinstance(exponent, int)
            for dimension, exponent in cls.unit_category.items()
        ):
            raise TypeError(
                f"{cls.__name__}.unit_category must map dimensions to integers"
            )
        if cls.variable_type not in ("continuous", "binary", "integer"):
            raise TypeError(f"{cls.__name__}.variable_type is invalid")
        if cls.index_relations != "auto" and not (
            isinstance(cls.index_relations, tuple)
            and all(
                isinstance(relation, type) and issubclass(relation, GenericRelationSet)
                for relation in cls.index_relations
            )
        ):
            raise TypeError(
                f"{cls.__name__}.index_relations must be 'auto' or a tuple "
                "of GenericRelationSet classes"
            )
        if cls.mask_default_value not in (None, 0.0):
            raise TypeError(
                f"{cls.__name__}.mask_default_value currently supports only 0.0"
            )
        if "build" in cls.__dict__:
            raise TypeError(f"{cls.__name__} must not override GenericVariable.build()")
        inherited_masks = []
        for base in cls.__bases__:
            inherited_masks.extend(getattr(base, "masks", ()))
        own_masks = cls.__dict__.get("masks", ())
        cls.masks = tuple(dict.fromkeys([*inherited_masks, *own_masks]))
        if not all(
            isinstance(mask, type) and issubclass(mask, GenericMask)
            for mask in cls.masks
        ):
            raise TypeError(f"{cls.__name__}.masks must contain GenericMask classes")
        if cls.masks and cls.mask_default_value is None:
            raise TypeError(
                f"{cls.__name__} must declare mask_default_value when using masks"
            )
        if not cls.masks and cls.mask_default_value is not None:
            raise TypeError(
                f"{cls.__name__} declares mask_default_value without any masks"
            )

    @classmethod
    @final
    def build(cls, model_constructor: "ModelConstructor") -> None:
        """Construct and register this variable on the optimization model."""
        if not cls.should_construct(model_constructor):
            return
        resolved = model_constructor.optimization_model.sets.resolve_indices(
            cls.indices,
            index_relations=cls.index_relations,
        )
        index_sets = (resolved.values, list(resolved.dimensions))
        declared_mask = model_constructor.optimization_model.masks.combine_for_indices(
            cls.masks,
            resolved.structural_mask,
        )
        mask = declared_mask
        if mask is not None:
            model_constructor.optimization_model.masks.log_reduction(cls.name, mask)
        model_constructor.optimization_model.add_variable(
            name=cls.name,
            index_sets=index_sets,
            variable_type=cls.variable_type,
            bounds=cls.get_bounds(model_constructor, index_sets),
            mask=mask,
            structural_mask=resolved.structural_mask,
            mask_default_value=(cls.mask_default_value if mask is not None else None),
            applied_relations=resolved.applied_relations,
            doc=cls.doc,
            unit_category=cls.unit_category,
        )

    @classmethod
    def get_bounds(
        cls,
        model_constructor: "ModelConstructor",
        index_sets: VariableIndexSet,
    ) -> VariableBounds:
        """Return the variable bounds, if any."""
        return cls.bounds

    @classmethod
    def should_construct(cls, model_constructor: "ModelConstructor") -> bool:
        """Return whether this variable is required by the current configuration."""
        return True
