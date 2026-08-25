from __future__ import annotations

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
    # Named strategy used by ParameterInputLoader. The strategy describes the
    # physical input layout; it is deliberately separate from model construction.
    input_loader: ClassVar[str] = "standard"
    input_name: ClassVar[str | None] = None
    input_indices: ClassVar[tuple[str, ...] | None] = None

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
    def construction_order(
        cls, parameters: list[type[GenericParameter]]
    ) -> list[type[GenericComputedParameters]]:
        """Return computed parameters in topological construction order."""
        computed = [parameter for parameter in parameters if issubclass(parameter, cls)]
        all_names = {parameter.name for parameter in parameters}
        for parameter in computed:
            missing = set(parameter.dependencies).difference(all_names)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(
                    f"Computed parameter {parameter.name!r} has unknown "
                    f"dependencies: {names}"
                )

        remaining = list(computed)
        computed_names = {parameter.name for parameter in computed}
        completed = all_names.difference(computed_names)
        ordered: list[type[GenericComputedParameters]] = []
        while remaining:
            ready = [
                parameter
                for parameter in remaining
                if set(parameter.dependencies).issubset(completed)
            ]
            if not ready:
                cycle = ", ".join(parameter.name for parameter in remaining)
                raise ValueError(f"Cyclic computed-parameter dependencies: {cycle}")
            for parameter in ready:
                remaining.remove(parameter)
                ordered.append(parameter)
                completed.add(parameter.name)
        return ordered
