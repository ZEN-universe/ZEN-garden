from abc import ABC
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from zen_garden.elements.model_constructor import ModelConstructor


class GenericVariable(ABC):
    """Abstract base class for variables in ZEN-garden."""

    name: ClassVar[str]
    indices: ClassVar[list[str]]
    doc: ClassVar[str]
    unit_category: ClassVar[dict[str, int]]
    integer: ClassVar[bool] = False
    binary: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        required = ("name", "indices", "doc", "unit_category")

        for attr in required:
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__name__} must define {attr!r}")

    @classmethod
    def get_index_sets(cls, model_constructor: "ModelConstructor"):
        """Return the model indices used to construct this variable."""
        return model_constructor.create_custom_set(cls.indices)

    @classmethod
    def get_bounds(cls, model_constructor: "ModelConstructor", index_sets):
        """Return the variable bounds, if any."""
        return None

    @classmethod
    def get_mask(cls, model_constructor: "ModelConstructor", index_sets):
        """Return an optional mask restricting the constructed variable."""
        return None

    @classmethod
    def should_construct(cls, model_constructor: "ModelConstructor") -> bool:
        """Return whether this variable is required by the current configuration."""
        return True
