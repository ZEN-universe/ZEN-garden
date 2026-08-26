from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from zen_garden.services.parameter_input_loader import ParameterInputLoader


class GenericParameter(ABC):
    """Abstract base class for parameters in ZEN-garden."""

    name: ClassVar[str]
    indices: ClassVar[tuple[str, ...]]
    doc: ClassVar[str]
    unit_category: ClassVar[dict[str, int]]
    time_series: ClassVar[bool] = False
    capacity_types: ClassVar[bool] = False
    set_time_steps: ClassVar[str | None] = None
    # Named strategy used by ParameterInputLoader. The strategy describes the
    # physical input layout; it is deliberately separate from model construction.
    input_loader: ClassVar[str] = "standard"
    input_name: ClassVar[str | None] = None
    input_indices: ClassVar[tuple[str, ...] | None] = None
    dependencies: ClassVar[list[str]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls.__dict__.get("_abstract_parameter", False):
            return

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

    @classmethod
    def construction_order(
        cls, parameters: list[type[GenericParameter]]
    ) -> list[type[GenericParameter]]:
        """Return all parameter specifications in global dependency order."""
        parameters_by_name: dict[str, type[GenericParameter]] = {}
        for parameter in parameters:
            existing = parameters_by_name.get(parameter.name)
            if existing is not None and existing is not parameter:
                raise ValueError(
                    f"Multiple parameter specifications define {parameter.name!r}: "
                    f"{existing.__name__} and {parameter.__name__}"
                )
            parameters_by_name[parameter.name] = parameter

        all_names = set(parameters_by_name)
        for parameter in parameters_by_name.values():
            missing = set(parameter.dependencies).difference(all_names)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(
                    f"Parameter {parameter.name!r} has unknown dependencies: {names}"
                )

        remaining = list(parameters_by_name.values())
        completed: set[str] = set()
        ordered: list[type[GenericParameter]] = []
        while remaining:
            ready = [
                parameter
                for parameter in remaining
                if set(parameter.dependencies).issubset(completed)
            ]
            if not ready:
                cycle = ", ".join(parameter.name for parameter in remaining)
                raise ValueError(f"Cyclic parameter dependencies: {cycle}")
            for parameter in ready:
                remaining.remove(parameter)
                ordered.append(parameter)
                completed.add(parameter.name)
        return ordered


class GenericComputedParameters(GenericParameter):
    """Parameter that is processed after ordinary input parameters.

    Dependencies are parameter names and define a directed acyclic graph used
    to determine processing order. Subclasses must declare the list explicitly,
    including an empty list when no other parameter is required.
    """

    _abstract_parameter = True
    dependencies: ClassVar[list[str]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "dependencies" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define 'dependencies'")
        if not isinstance(cls.dependencies, list):
            raise TypeError(f"{cls.__name__}.dependencies must be a list")

    @classmethod
    @abstractmethod
    def store_input_data(cls, element: Any, loader: ParameterInputLoader) -> None:
        """Load or calculate the parameter and store it on ``element``."""
