from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor


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
        component_data, index_list, units = cls.get_model_data(
            model_constructor, index_names
        )
        component_data = cls._ensure_multi_index(component_data)
        model_constructor.zen_model.add_parameter(
            cls.name,
            cls.doc,
            (component_data, index_list),
            units,
        )

    @classmethod
    def get_model_data(
        cls, model_constructor: "ModelConstructor", index_names: list[str]
    ):
        """Collect this parameter's stored values, model indices, and units."""
        if model_constructor.element_class is type(model_constructor.energy_system):
            component_data = getattr(model_constructor.energy_system, cls.name)
            index_list = index_names
            if cls.set_time_steps is not None:
                component_data = component_data[
                    model_constructor.zen_model.sets[cls.set_time_steps]
                ]
            else:
                if not isinstance(component_data, float):
                    component_data = component_data.squeeze()
            units = model_constructor.energy_system.units.get(cls.name, {})
            return component_data, index_list, units

        custom_set, index_list = model_constructor.create_custom_set(index_names)
        component_data, units, attribute_is_series = (
            model_constructor.element_registry.get_attribute_of_all_elements_with_units(
                model_constructor.element_class,
                cls.name,
                capacity_types=cls.capacity_types,
            )
        )
        if np.size(custom_set):
            if attribute_is_series:
                component_data = pd.concat(component_data, keys=component_data.keys())
            else:
                component_data = pd.Series(component_data)
            component_data = cls._select_model_index(component_data, custom_set)
        return component_data, index_list, units

    @staticmethod
    def _select_model_index(component_data, custom_set):
        """Restrict parameter data to its model index."""
        try:
            if len(component_data) == len(custom_set) and len(custom_set[0]) == len(
                component_data.index[0]
            ):
                return component_data
            return component_data[custom_set]
        except Exception:
            custom_index = pd.Index(custom_set)
            reduced_index = custom_index.copy()
            assert isinstance(custom_index, pd.MultiIndex), (
                f"Custom set {custom_set} is not a MultiIndex. "
                "Please check the index sets of the component."
            )
            for level, shape in enumerate(custom_index.levshape):
                if shape == 1:
                    reduced_index = reduced_index.droplevel(level)
            try:
                component_data = component_data[reduced_index]
                component_data.index = custom_index
                return component_data
            except KeyError as err:
                raise KeyError(
                    f"the custom set {custom_set} cannot be used as a subindex of "
                    f"{component_data.index}"
                ) from err

    @staticmethod
    def _ensure_multi_index(component_data):
        """Represent a one-dimensional Series index as a MultiIndex."""
        if isinstance(component_data, pd.Series) and not isinstance(
            component_data.index, pd.MultiIndex
        ):
            component_data.index = pd.MultiIndex.from_product(
                [component_data.index.to_list()]
            )
        return component_data

    @classmethod
    def store_input_data(cls, element: Any) -> None:
        """Load and store a parameter using the standard input layout."""
        name = cls.input_name or cls.name
        indices = cls._input_indices(element)
        value = element.element_data_loader.extract_input_data(
            name,
            index_sets=indices,
            unit_category=cls.unit_category,
        )
        cls._store_value(element, cls.name, value)

        if cls.capacity_types and cls._has_energy_capacity(element):
            energy_units = dict(cls.unit_category)
            energy_units.pop("time", None)
            energy_value = element.element_data_loader.extract_input_data(
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
