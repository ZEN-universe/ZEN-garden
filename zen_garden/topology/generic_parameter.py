from abc import ABC
from typing import ClassVar


class GenericParameter(ABC):
    """Abstract base class for parameters in ZEN-garden."""

    name: ClassVar[str]
    indices: ClassVar[tuple[str, ...]]
    doc: ClassVar[str]
    unit_category: ClassVar[dict[str, int]]
    time_series: ClassVar[bool] = False
    capacity_types: ClassVar[bool] = False
    set_time_steps: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        required = ("name", "indices", "doc", "unit_category")

        for attr in required:
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__name__} must define {attr!r}")

    # This is a classmethod so that it can be called without creating an
    # instance of the class, e.g. Parameter.build() rather than Parameter().build().
    @classmethod
    def build(cls):
        """Build the parameter."""
        raise NotImplementedError("ToDO:")
