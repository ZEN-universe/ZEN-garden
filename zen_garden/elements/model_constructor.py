"""Abstract constructor for elements.

All subclasses of ModelConstructor must implement the abstract methods to construct
the sets, parameters, variables, and constraints for their respective elements.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from zen_garden.elements.element import Element
from zen_garden.services.service_container import ServiceContainer
from zen_garden.topology.generic_constraint import GenericConstraint
from zen_garden.topology.generic_set import GenericSet

if TYPE_CHECKING:
    from zen_garden.elements.energy_system import EnergySystem
    from zen_garden.model.config import Config
    from zen_garden.model.time_steps import TimeStepsDicts
    from zen_garden.model.zen_model import ZenModel
    from zen_garden.services.element_registry import ElementRegistry

logger = logging.getLogger(__name__)


class ModelConstructor(ABC):
    element_class: ClassVar[type["Element"] | type["EnergySystem"]] = Element
    constraints: list[type[GenericConstraint]] = []
    sets: list[type[GenericSet]] = []

    def __init__(
        self,
        service_container: "ServiceContainer",
        config: "Config",
        element_registry: "ElementRegistry",
        zen_model: "ZenModel",
        energy_system: "EnergySystem",
        time_steps: "TimeStepsDicts",
    ):
        self.service_container = service_container
        self.config = config
        self.element_registry = element_registry
        self.zen_model = zen_model
        self.energy_system = energy_system
        self.time_steps = time_steps

    @abstractmethod
    def has_elements(self) -> bool:
        """Checks if the element has any elements to construct.

        :return: True if the element has elements, False otherwise
        """
        pass

    def construct_sets(self):
        """Constructs the Sets of this class."""
        logger.info(f"Constructing sets for {self.element_class.__name__}")
        for model_set in self.sets:
            model_set.build(self)

    def construct_params(self):
        logger.info(f"Constructing parameters for {self.element_class.name}")

        for parameter in self.parameters:
            # rename time steps
            index_names = [
                "set_time_steps_operation" if x == "set_hours" else x
                for x in parameter.indices
            ]
            self.add_parameter(
                name=parameter.name,
                index_names=index_names,
                doc=parameter.doc,
                capacity_types=parameter.capacity_types,
                set_time_steps=parameter.set_time_steps,
            )

    @abstractmethod
    def construct_vars(self):
        """Constructs the Vars of this class."""
        pass

    def construct_expressions(self):  # noqa: B027
        """Construct reusable expressions from parameters and variables."""
        pass

    def construct_constraints(self):
        """Constructs the Constraints of this class."""
        logger.info(f"Constructing constraints for {self.element_class.__name__}")

        for ConstraintClass in self.constraints:
            self.service_container.build(ConstraintClass).build()

    def construct_objective(self):  # noqa: B027
        """Constructs the Objective of this class."""
        # do nothing by default, only overriden by EnergySystemConstructor
        pass

    def create_custom_set(self, list_index: list[str]):
        """Creates custom set for model component. See
        :meth:`zen_garden.model.components.set_registry.SetRegistry.create_custom_set`.

        :param list_index: list of names of indices
        :return: list_index: list of names of indices
        """
        return self.zen_model.create_custom_set(list_index, self.element_class)

    def add_parameter(
        self,
        name: str,
        doc: str,
        index_names: list[str] | None = None,
        capacity_types: bool = False,
        set_time_steps: str | None = None,
    ):
        """Adds a parameter to the optimization model for components without data.

        :param name: name of parameter
        :param doc: docstring of parameter
        :param index_names: list of names of index sets
        :param capacity_types: boolean if extracted for capacities
        """
        component_data, index_list, dict_of_units = self._initialize_component(
            name, index_names, capacity_types, set_time_steps
        )
        component_data = self._ensure_pd_series_multi_index(component_data)
        data = component_data, index_list

        self.zen_model.add_parameter(name, doc, data, dict_of_units)

    def _initialize_component(
        self,
        component_name: str,
        index_names: list[str] | None,
        capacity_types: bool = False,
        set_time_steps: str | None = None,
    ):
        """Initialize a modeling component by extracting the stored input data.

        Args:
            component_name: name of the modeling component
            index_names: names of index sets
            capacity_types: boolean if extracted for capacities
            set_time_steps: name of the set of time steps to extract data for

        Returns:
            component_data: extracted data as pd.Series
            index_list: list of names of index sets
            dict_of_units: dictionary of units for the component
        """

        if self.element_class.__name__ == "EnergySystem":
            component_data = getattr(self.energy_system, component_name)
            if set_time_steps is not None:
                index_list = [set_time_steps]
            else:
                index_list = []
            return (
                component_data,
                index_list,
                self.energy_system.units.get(component_name, {}),
            )

        if index_names is None:
            raise ValueError(f"Index names for {component_name} not specified")
        custom_set, index_list = self.create_custom_set(index_names)
        component_data, dict_of_units, attribute_is_series = (
            self.element_registry.get_attribute_of_all_elements_with_units(
                self.element_class,
                component_name,
                capacity_types=capacity_types,
            )
        )
        if np.size(custom_set):
            if attribute_is_series:
                component_data = pd.concat(component_data, keys=component_data.keys())
            else:
                component_data = pd.Series(component_data)
            component_data = self._check_for_subindex(component_data, custom_set)

        return component_data, index_list, dict_of_units

    def _check_for_subindex(self, component_data, custom_set):
        """Check if the custom_set can be a subindex of component_data.

        returns subindexed component_data.

        :param component_data: extracted data as pd.Series
        :param custom_set: custom set as subindex of component_data
        :return: component_data: extracted subindexed data as pd.Series
        """
        # if custom_set is subindex of component_data, return subset of component_data
        try:
            if len(component_data) == len(custom_set) and len(custom_set[0]) == len(
                component_data.index[0]
            ):
                return component_data
            else:
                return component_data[custom_set]
        # else delete trivial index levels (that have a single value) and try again
        except Exception:
            _custom_index = pd.Index(custom_set)
            _reduced_custom_index = _custom_index.copy()
            assert isinstance(_custom_index, pd.MultiIndex), (
                f"Custom set {custom_set} is not a MultiIndex. "
                f"Please check the index sets of the component."
            )
            for _level, _shape in enumerate(_custom_index.levshape):
                if _shape == 1:
                    _reduced_custom_index = _reduced_custom_index.droplevel(_level)
            try:
                component_data = component_data[_reduced_custom_index]
                component_data.index = _custom_index
                return component_data
            except KeyError as err:
                raise KeyError(
                    f"the custom set {custom_set} cannot be used as a subindex of "
                    f"{component_data.index}"
                ) from err

    def _ensure_pd_series_multi_index(self, component_data):
        """Convert pd.Series index to pd.MultiIndex.

        :param component_data: extracted data as pd.Series
        :return: component_data: extracted data as pd.Series with MultiIndex
        """
        if isinstance(component_data, pd.Series) and not isinstance(
            component_data.index, pd.MultiIndex
        ):
            component_data.index = pd.MultiIndex.from_product(
                [component_data.index.to_list()]
            )
        return component_data
