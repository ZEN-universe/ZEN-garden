"""Declarative model-domain and relation-set specifications."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias, final

from zen_garden.model.component_types.component import GenericComponent
from zen_garden.model.registries.set import DomainSet, RelationSet

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor


class GenericDomainSet(GenericComponent, ABC):
    """Declarative specification for one independent model domain."""

    name: ClassVar[str]
    symbol: ClassVar[str]
    doc: ClassVar[str]
    subset_of: ClassVar[type[GenericDomainSet] | None] = None
    _abstract_component = True

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("_abstract_component", False):
            return
        symbol = cls.__dict__.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise TypeError(f"{cls.__name__}.symbol must be a non-empty string")
        if cls.subset_of is not None and not (
            isinstance(cls.subset_of, type)
            and issubclass(cls.subset_of, GenericDomainSet)
        ):
            raise TypeError(
                f"{cls.__name__}.subset_of must be a GenericDomainSet class"
            )
        if "build" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override GenericDomainSet.build()"
            )

    @classmethod
    @abstractmethod
    def get_data(cls, model_constructor: "ModelConstructor") -> Iterable[Any]:
        """Extract the ordered members of this domain."""

    @classmethod
    @final
    def build(cls, model_constructor: "ModelConstructor") -> None:
        """Extract, validate, and register this domain."""
        registry = model_constructor.optimization_model.sets
        data = cls.get_data(model_constructor)
        if isinstance(data, Mapping):
            raise TypeError(
                f"Domain set {cls.name!r} returned mapping data; use a "
                "GenericRelationSet for tuple relationships"
            )
        registry.add_domain(
            DomainSet(cls.name, data, cls.doc),
            subset_of=cls.subset_of,
        )


class GenericRelationSet(GenericComponent, ABC):
    """Declarative specification for one sparse relation between domains."""

    name: ClassVar[str]
    doc: ClassVar[str]
    dimensions: ClassVar[tuple[type[GenericDomainSet], ...]]
    apply_by_default: ClassVar[bool] = True
    _abstract_component = True

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("_abstract_component", False):
            return
        dimensions = cls.__dict__.get("dimensions")
        if not isinstance(dimensions, tuple) or len(dimensions) < 2:
            raise TypeError(
                f"{cls.__name__}.dimensions must contain at least two domains"
            )
        if not all(
            isinstance(dimension, type) and issubclass(dimension, GenericDomainSet)
            for dimension in dimensions
        ):
            raise TypeError(
                f"{cls.__name__}.dimensions must contain GenericDomainSet classes"
            )
        if len(set(dimensions)) != len(dimensions):
            raise TypeError(f"{cls.__name__}.dimensions must be unique")
        if not isinstance(cls.apply_by_default, bool):
            raise TypeError(f"{cls.__name__}.apply_by_default must be a bool")
        if "build" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override GenericRelationSet.build()"
            )

    @classmethod
    @abstractmethod
    def get_combinations(cls, model_constructor: "ModelConstructor") -> Any:
        """Extract ordered relation tuples or a two-dimensional mapping."""

    @classmethod
    @final
    def build(cls, model_constructor: "ModelConstructor") -> None:
        """Extract, validate, and register this relation."""
        model_constructor.optimization_model.sets.add_relation(
            RelationSet(
                cls.name,
                tuple(dimension.name for dimension in cls.dimensions),
                cls.get_combinations(model_constructor),
                cls.doc,
                cls.apply_by_default,
            )
        )


SetDeclaration: TypeAlias = type[GenericDomainSet] | type[GenericRelationSet]
