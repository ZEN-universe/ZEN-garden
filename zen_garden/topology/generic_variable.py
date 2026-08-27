from abc import ABC
from typing import ClassVar


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
    def build(cls):
        """Build the Variable."""
        raise NotImplementedError("ToDO:")

    @classmethod
    def get_bounds(cls, *args, **kwargs):
        """Build the Variable."""
        raise NotImplementedError("ToDO:")
