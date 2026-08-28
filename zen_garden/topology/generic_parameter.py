from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from zen_garden.elements.model_constructor import ModelConstructor


class GenericParameter(ABC):
    """Abstract base class for parameters in ZEN-garden.

    Lifecycle:

    * :meth:`store_input_data` runs once during preprocessing and puts the
      parameter's values onto the elements (usually read from input files).
    * :meth:`build` runs during model construction -- once per rolling-horizon
      step -- and registers the parameter on the optimization model. The default
      reads the values stored by :meth:`store_input_data`; parameters that are
      *derived* from other model parameters override :meth:`build` instead (and
      leave :meth:`store_input_data` empty).
    """

    name: ClassVar[str]
    indices: ClassVar[tuple[str, ...]]
    doc: ClassVar[str]
    unit_category: ClassVar[dict[str, int]]
    time_series: ClassVar[bool] = False
    capacity_types: ClassVar[bool] = False
    set_time_steps: ClassVar[str | None] = None
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
        if not isinstance(cls.dependencies, list):
            raise TypeError(f"{cls.__name__}.dependencies must be a list")

    @classmethod
    def build(cls, model_constructor: "ModelConstructor") -> None:
        """Register this parameter on the optimization model.

        The default registers the values put on the elements by
        :meth:`store_input_data`. Derived parameters override this to compute
        their data from other, already-registered model parameters.
        """
        index_names = [
            "set_time_steps_operation" if index == "set_hours" else index
            for index in cls.indices
        ]
        model_constructor.add_parameter(
            name=cls.name,
            index_names=index_names,
            doc=cls.doc,
            capacity_types=cls.capacity_types,
            set_time_steps=cls.set_time_steps,
        )

    @classmethod
    def store_input_data(cls, element: Any) -> None:
        """Load and store a parameter using the standard input layout."""
        name = cls.input_name or cls.name
        indices = cls._input_indices(element)
        value = element.data_input.extract_input_data(
            name,
            index_sets=indices,
            unit_category=cls.unit_category,
        )
        cls._store_value(element, cls.name, value)

        if cls.capacity_types and cls._has_energy_capacity(element):
            energy_units = dict(cls.unit_category)
            energy_units.pop("time", None)
            energy_value = element.data_input.extract_input_data(
                f"{name}_energy",
                index_sets=indices,
                unit_category=energy_units,
            )
            cls._store_value(element, f"{cls.name}_energy", energy_value)

    @classmethod
    def _store_value(cls, element: Any, name: str, value: Any) -> None:
        """Store a loaded value in its time-series or scalar destination."""
        if cls.time_series:
            element.raw_time_series[name] = value
        else:
            setattr(element, name, value)

    @classmethod
    def _input_indices(cls, element: Any) -> list[str]:
        """Resolve schema indices to the physical input indices for an element."""
        if cls.input_indices is not None:
            indices = list(cls.input_indices)
        else:
            owner_labels = {
                base.__dict__["label"]
                for base in type(element).mro()
                if "label" in base.__dict__
            }
            indices = [
                index
                for index in cls.indices
                if index not in owner_labels and index != "set_capacity_types"
            ]

        location_type = getattr(element, "location_type", None)
        if "set_location" in indices and location_type is None:
            raise ValueError(f"Element {element.name!r} has no location type")
        return [
            str(location_type) if index == "set_location" else index
            for index in indices
        ]

    @staticmethod
    def _has_energy_capacity(element: Any) -> bool:
        return element.name in element.config.system.set_storage_technologies

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
