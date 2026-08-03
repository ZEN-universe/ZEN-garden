"""Abstract constructor for elements.

All subclasses of ElementConstructor must implement the abstract methods to construct
the sets, parameters, variables, and constraints for their respective elements.
"""

import itertools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from zen_garden.elements.element import Element

if TYPE_CHECKING:
    from zen_garden.elements.energy_system import EnergySystem
    from zen_garden.model.components.zen_set import ZenSet
    from zen_garden.model.config import Config
    from zen_garden.model.time_steps import TimeStepsDicts
    from zen_garden.model.zen_model import ZenModel
    from zen_garden.services.element_registry import ElementRegistry


class ElementConstructor(ABC):
    element_class: type[Element] = Element

    def __init__(
        self,
        config: "Config",
        element_registry: "ElementRegistry",
        zen_model: "ZenModel",
        energy_system: "EnergySystem",
        time_steps: "TimeStepsDicts",
    ):
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

    @abstractmethod
    def construct_sets(self):
        """Constructs the Sets of this class."""
        pass

    @abstractmethod
    def construct_params(self):
        """Constructs the Params of this class."""
        pass

    @abstractmethod
    def construct_vars(self):
        """Constructs the Vars of this class."""
        pass

    @abstractmethod
    def construct_constraints(self):
        """Constructs the Constraints of this class."""
        pass

    def construct_objective(self):  # noqa: B027
        """Constructs the Objective of this class."""
        # do nothing by default, only overriden by EnergySystemConstructor
        pass

    def create_custom_set(self, list_index: list[str]):
        """Creates custom set for model component.

        :param list_index: list of names of indices
        :return: list_index: list of names of indices
        """
        list_index = list(list_index)  # make a copy of the list to avoid side effects

        # Case 1: all index sets are already defined in model and no set is indexed
        if all(
            index in self.zen_model.sets.sets
            and not self.zen_model.sets.is_indexed(index)
            for index in list_index
        ):
            list_sets = [
                self.zen_model.sets[index]
                for index in list_index
                if index in self.zen_model.sets
            ]
            # return indices as cartesian product of sets
            custom_set: list[tuple[ZenSet, ...]] | list[ZenSet] = (
                list(itertools.product(*list_sets))
                if len(list_sets) > 1
                else list(list_sets[0])
            )
            return custom_set, list_index

        if list_index[0] not in self.zen_model.indexing_sets:
            raise NotImplementedError(
                f"Index <{list_index[0]}> is not in the indexing sets."
            )

        # Case 2: first index is indexed, build custom set based on first index
        custom_set = []
        for element in self.zen_model.sets[list_index[0]]:
            append_element = True
            list_sets = []

            for index in list_index[1:]:
                # if the set already exist in model
                if index in self.zen_model.sets:
                    append = self._handle_existing_set(index, element, list_sets)
                    if not append:
                        raise NotImplementedError(
                            f"Index <{index}> is not known in sets."
                        )
                    continue

                # if index is set_location
                if index == "set_location":
                    self._handle_set_location_index(element, list_sets)
                    continue

                # if set is built for pwa capex:
                if "set_capex" in index:
                    append_element = self._append_set_capex_index(element, index)
                    continue

                # if set is used to determine if on-off behavior is modeled
                # exclude technologies which have no min_load
                if "on_off" in index:
                    append_element = self._append_on_off_modeled(element, index)
                    continue

                # split in capacity types of power and energy
                if index == "set_capacity_types":
                    self._handle_set_capacity_types_index(element, list_sets)
                    continue

                raise NotImplementedError(f"Index <{index}> not known")

            # append indices to custom_set if element is supposed to be appended
            if append_element:
                if list_sets:
                    custom_set.extend(list(itertools.product([element], *list_sets)))
                else:
                    custom_set.extend([element])
        return custom_set, list_index

    def _handle_existing_set(
        self, index: str, element: "ZenSet", list_sets: "list[ZenSet]"
    ):
        """Handles existing sets in the model.
        Returns True if handled, False if unknown.

        :param index: index to handle
        :param element: element to handle
        :param sets: sets of the optimization setup
        :param list_sets: list of sets to append
        """
        if not self.zen_model.sets.is_indexed(index):
            list_sets.append(self.zen_model.sets[index])
            return True
        elif self.zen_model.sets.get_index_name(index) in self.zen_model.sets.sets:
            list_sets.append(self.zen_model.sets[index][element])
            return True
        return False

    def _append_set_capex_index(self, element: str, index: str) -> bool:
        """Checks if the capex of a technology needs to be modeled as pwa or linear.

        :param element: technology in model
        :param index: index to check
        :return model_capex: Bool indicating if capex must be modeled as pwa or linear
        """
        if element not in self.zen_model.sets["set_conversion_technologies"]:
            return False

        capex_is_pwa = self.element_registry.get_attribute_of_specific_element(
            self.element_class, element, "capex_is_pwa"
        )
        return not (
            ("linear" in index and capex_is_pwa)
            or ("pwa" in index and not capex_is_pwa)
        )

    def _append_on_off_modeled(self, element: str, index: str) -> bool:
        """Checks if the on-off-behavior (min-load) of a technology needs to be modeled.

        :param element: technology in model
        :param index: index to check
        :return model_on_off: Bool indicating if on-off-behavior needs to be modeled
        """
        model_on_off = self.check_on_off_modeled(element)
        return not (("set_no_on_off" in index and model_on_off) or (not model_on_off))

    def _handle_set_location_index(self, element: str, list_sets: "list[ZenSet]"):
        """Handles the set_location index for the custom set.

        :param element: element to handle
        :param sets: sets of the optimization setup
        :param list_sets: list of sets to append
        """
        if (
            element in self.zen_model.sets["set_conversion_technologies"]
            or element in self.zen_model.sets["set_storage_technologies"]
            or element in self.zen_model.sets["set_retrofitting_technologies"]
        ):
            list_sets.append(self.zen_model.sets["set_nodes"])
        elif element in self.zen_model.sets["set_transport_technologies"]:
            list_sets.append(self.zen_model.sets["set_edges"])

    def _handle_set_capacity_types_index(self, element: str, list_sets: list[str]):
        """Handles the set_capacity_types index for the custom set.

        :param element: element to handle
        :param sets: sets of the optimization setup
        :param list_sets: list of sets to append
        """
        if element in self.zen_model.sets["set_storage_technologies"]:
            list_sets.append(self.config.system.set_capacity_types)
        else:
            list_sets.append([self.config.system.set_capacity_types[0]])

    def check_on_off_modeled(self, tech: str):
        """Classmethod checks if on-off-behavior of a technology needs to be modeled.

        If the technology has a minimum load of 0 for all nodes and time steps, and all
        dependent carriers have a lower bound of 0 (only for conversion technologies
        modeled as pwa), then on-off-behavior is not necessary to model.

        :param tech: technology in model
        :return model_on_off: Bool indicating if on-off-behaviour needs to be modeled
        """
        # check if any min load
        unique_min_load = list(
            set(
                self.element_registry.get_attribute_of_specific_element(
                    self.element_class, tech, "min_load"
                ).values
            )
        )
        # disable if only one unique min_load which is zero
        return not (len(unique_min_load) == 1 and unique_min_load[0] == 0)

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
