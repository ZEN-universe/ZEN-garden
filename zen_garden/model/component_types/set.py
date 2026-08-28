"""Declarative model-set specifications."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor


class GenericSet(ABC):
    """Base class for model sets constructed from schema specifications."""

    name: ClassVar[str]
    doc: ClassVar[str]
    index_set: ClassVar[str | None] = None
    indexing_set: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for attribute in ("name", "doc"):
            if not hasattr(cls, attribute):
                raise TypeError(f"{cls.__name__} must define {attribute!r}")

    @classmethod
    @abstractmethod
    def get_data(cls, model_constructor: "ModelConstructor") -> Any:
        """Extract the members of this set from model-construction state."""

    @classmethod
    def build(cls, model_constructor: "ModelConstructor") -> None:
        """Extract and register this set on the optimization model."""
        model_constructor.optimization_model.add_set(
            name=cls.name,
            data=cls.get_data(model_constructor),
            doc=cls.doc,
            index_set=cls.index_set,
        )
        if cls.indexing_set:
            model_constructor.optimization_model.indexing_sets.append(cls.name)
