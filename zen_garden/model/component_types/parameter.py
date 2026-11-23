from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, final

import numpy as np
import pandas as pd
import xarray as xr

from zen_garden.model.component_types.component import GenericComponent
from zen_garden.model.component_types.set import GenericRelationSet
from zen_garden.model.linear_expression import validate_component_indices

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor
    from zen_garden.model.registries.set import ResolvedIndex


@dataclass(frozen=True, slots=True)
class ParameterInputData:
    """Values and units loaded for one parameter."""

    values: Any
    units: Any


@dataclass(frozen=True, slots=True)
class ParameterData:
    """Final parameter value registered in the optimization model."""

    values: Any
    indices: tuple[str, ...]
    units: Any = None
    resolved_index: ResolvedIndex | None = None


class GenericParameter(GenericComponent, ABC):
    """Abstract base class for parameters in ZEN-garden.

    Lifecycle:

    * :meth:`get_input_data` returns values during preprocessing for the
      framework to store on the elements.
    * :meth:`build` runs during model construction -- once per rolling-horizon
      step -- and registers the parameter on the optimization model. The default
      reads the stored values through :meth:`get_parameter_value`; derived
      parameters override that hook instead.
    """

    name: ClassVar[str]
    symbol: ClassVar[str]
    indices: ClassVar[tuple[str, ...]]
    doc: ClassVar[str]
    unit_category: ClassVar[dict[str, int]]
    time_series: ClassVar[bool] = False
    set_time_steps: ClassVar[str | None] = None
    input_name: ClassVar[str | None] = None
    input_indices: ClassVar[tuple[str, ...] | None] = None
    default_value: ClassVar[int | float | str | None] = None
    default_unit: ClassVar[str | None] = None
    dependencies: ClassVar[tuple[type["GenericParameter"], ...]] = ()
    index_relations: ClassVar[str | tuple[type[GenericRelationSet], ...]] = "auto"
    squeeze_singleton: ClassVar[bool] = True
    _abstract_component = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls.__dict__.get("_abstract_component", False):
            return

        required = ("symbol", "indices", "unit_category")

        for attr in required:
            if attr not in cls.__dict__:
                raise TypeError(f"{cls.__name__} must define {attr!r}")
        for attr in ("symbol",):
            if (
                not isinstance(getattr(cls, attr), str)
                or not getattr(cls, attr).strip()
            ):
                raise TypeError(f"{cls.__name__}.{attr} must be a non-empty string")
        if isinstance(cls.indices, str) or not all(
            isinstance(index, str) and index for index in cls.indices
        ):
            raise TypeError(f"{cls.__name__}.indices must contain non-empty set names")
        cls.indices = tuple(cls.indices)
        if not isinstance(cls.unit_category, dict) or not all(
            isinstance(dimension, str) and isinstance(exponent, int)
            for dimension, exponent in cls.unit_category.items()
        ):
            raise TypeError(
                f"{cls.__name__}.unit_category must map dimensions to integers"
            )
        if "build" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override GenericParameter.build()"
            )
        if cls.index_relations != "auto" and not (
            isinstance(cls.index_relations, tuple)
            and all(
                isinstance(relation, type) and issubclass(relation, GenericRelationSet)
                for relation in cls.index_relations
            )
        ):
            raise TypeError(
                f"{cls.__name__}.index_relations must be 'auto' or a tuple "
                "of GenericRelationSet classes"
            )
        if cls.default_value is not None and cls.default_value not in (0, 1, "inf"):
            raise TypeError(
                f"{cls.__name__}.default_value must be 0, 1, 'inf', or None"
            )
        if (cls.default_value is None) != (cls.default_unit is None):
            raise TypeError(
                f"{cls.__name__}.default_value and default_unit must be set together"
            )

    @classmethod
    @final
    def build(cls, model_constructor: "ModelConstructor") -> None:
        """Register this parameter on the optimization model.

        The default registers the value returned by
        :meth:`get_parameter_value`.
        """
        parameter = cls.get_parameter_value(model_constructor)
        resolved = parameter.resolved_index or (
            model_constructor.optimization_model.sets.resolve_indices(
                parameter.indices,
                index_relations=cls.index_relations,
            )
        )
        component_data = cls._ensure_multi_index(parameter.values)
        if isinstance(component_data, xr.DataArray):
            validate_component_indices(cls.name, parameter.indices, component_data)
            registry_data: xr.DataArray | tuple[Any, list[str]] = component_data
        else:
            registry_data = (component_data, list(parameter.indices))
        model_constructor.optimization_model.add_parameter(
            cls.name,
            cls.doc,
            registry_data,
            parameter.units,
            structural_mask=resolved.structural_mask,
            applied_relations=resolved.applied_relations,
        )

    @classmethod
    def get_parameter_value(
        cls, model_constructor: "ModelConstructor"
    ) -> ParameterData:
        """Collect this parameter's stored values, model indices, and units."""
        index_names = list(cls.indices)
        if model_constructor.element_class is type(model_constructor.energy_system):
            resolved = model_constructor.optimization_model.sets.resolve_indices(
                index_names,
                index_relations=cls.index_relations,
            )
            component_data = getattr(model_constructor.energy_system, cls.name)
            index_list = index_names
            if cls.set_time_steps is not None:
                component_data = component_data[
                    model_constructor.optimization_model.sets[cls.set_time_steps]
                ]
            else:
                if cls.squeeze_singleton and not isinstance(component_data, float):
                    component_data = component_data.squeeze()
            units = model_constructor.energy_system.units.get(cls.name, {})
            return ParameterData(
                component_data,
                tuple(index_list),
                units,
                resolved,
            )

        resolved = model_constructor.optimization_model.sets.resolve_indices(
            index_names,
            index_relations=cls.index_relations,
        )
        custom_set = resolved.values
        index_list = list(resolved.dimensions)
        component_data, units, attribute_is_series = (
            model_constructor.element_registry.get_attribute_of_all_elements_with_units(
                model_constructor.element_class,
                cls.name,
            )
        )
        if np.size(custom_set):
            if attribute_is_series:
                component_data = pd.concat(component_data, keys=component_data.keys())
            else:
                component_data = pd.Series(component_data)
            component_data = cls._select_model_index(component_data, custom_set)
        return ParameterData(
            component_data,
            tuple(index_list),
            units,
            resolved,
        )

    @classmethod
    def _select_model_index(cls, component_data, custom_set):
        """Restrict parameter data to its model index."""
        try:
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
                    f"Parameter {cls.name!r} data cannot be aligned to resolved "
                    f"index {custom_set!r}; source index is {component_data.index!r}"
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
    def get_input_data(cls, element: Any) -> ParameterInputData | None:
        """Return this parameter's values from the input dataset."""
        name = cls.input_name or cls.name
        indices = cls._input_indices(element)
        value = element.element_data_loader.extract_input_data(
            name,
            index_sets=indices,
            unit_category=cls.unit_category,
            parameter_default_value=cls.default_value,
            parameter_default_unit=cls.default_unit,
        )
        return ParameterInputData(value, element.units.get(name))

    @classmethod
    def load_input_data(cls, element: Any) -> None:
        """Store this parameter's returned input data on its owning element."""
        data = cls.get_input_data(element)
        if data is None:
            return
        cls._store_value(element, cls.name, data.values)
        if data.units is not None:
            element.units[cls.name] = data.units

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
            indices = [index for index in cls.indices if index not in owner_labels]

        location_type = getattr(element, "location_type", None)
        if "set_location" in indices and location_type is None:
            raise ValueError(f"Element {element.name!r} has no location type")
        indices = [
            str(location_type) if index == "set_location" else index
            for index in indices
        ]
        if "set_representative_hours" in indices:
            indices.remove("set_representative_hours")
            indices.remove("set_years")
            indices.append("set_hours")
        indices = [
            (
                "set_technologies_existing"
                if index == "set_existing_capacity_ids"
                else index
            )
            for index in indices
        ]
        return indices
