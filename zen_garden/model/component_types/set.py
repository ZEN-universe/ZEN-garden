"""Declarative model-set specifications."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, final

import pandas as pd

from zen_garden.model.component_types.component import GenericComponent
from zen_garden.model.component_types.index import GenericIndex
from zen_garden.model.registries.set import DomainSet, RelationSet

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor


class GenericSet(GenericComponent, ABC):
    """Base class for model sets constructed from schema specifications."""

    name: ClassVar[str]
    symbol: ClassVar[str]
    doc: ClassVar[str]
    index_set: ClassVar[type[GenericSet] | type[GenericIndex] | None] = None
    _abstract_component = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("_abstract_component", False):
            return
        symbol = cls.__dict__.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise TypeError(f"{cls.__name__}.symbol must be a non-empty string")
        if cls.index_set is not None and not (
            isinstance(cls.index_set, type)
            and issubclass(cls.index_set, (GenericSet, GenericIndex))
        ):
            raise TypeError(
                f"{cls.__name__}.index_set must be a GenericSet or GenericIndex class"
            )
        if "build" in cls.__dict__:
            raise TypeError(f"{cls.__name__} must not override GenericSet.build()")

    @classmethod
    @abstractmethod
    def get_data(cls, model_constructor: "ModelConstructor") -> Any:
        """Extract the members of this set from model-construction state."""

    @classmethod
    @final
    def build(cls, model_constructor: "ModelConstructor") -> None:
        """Extract and register this set on the optimization model."""
        registry = model_constructor.optimization_model.sets
        data = cls.get_data(model_constructor)
        if not isinstance(data, Mapping):
            if cls.index_set is not None:
                raise ValueError(
                    f"Set {cls.name!r} declares an index domain but returned "
                    "non-mapping data"
                )
            registry.add_domain(DomainSet(cls.name, data, cls.doc))
            return

        if cls.index_set is None:
            raise ValueError(f"Indexed set {cls.name!r} requires an index domain")
        parent_name = cls.index_set.name
        flattened = [member for values in data.values() for member in values]
        members = pd.Index(flattened, tupleize_cols=False).drop_duplicates()
        registry.add_domain(DomainSet(cls.name, members, cls.doc))
        relation_name = (
            f"relation_{parent_name.removeprefix('set_')}_"
            f"{cls.name.removeprefix('set_')}"
        )
        registry.add_relation(
            RelationSet(
                relation_name,
                (parent_name, cls.name),
                data,
                cls.doc,
            )
        )
        registry.legacy_indexed_set_relations[cls.name] = relation_name
        registry.docs[cls.name] = registry.compile_doc_string(
            cls.doc,
            name=cls.name,
            index_list=[parent_name],
        )
