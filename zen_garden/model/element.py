"""Class defining a standard Element.
Contains methods to add parameters, variables and constraints to the
optimization problem. Parent class of the Carrier and Technology classes.
The class takes the concrete optimization model as an input.
"""

import itertools
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import psutil

from zen_garden.model.components.index_set import IndexSet
from zen_garden.model.components.zen_set import ZenSet
from zen_garden.model.config import Config
from zen_garden.model.context import Context
from zen_garden.model.zen_model import ZenModel
from zen_garden.preprocess.extract_input_data import DataInput

if TYPE_CHECKING:
    from zen_garden.model.energy_system import EnergySystem
    from zen_garden.preprocess.unit_handling import UnitHandling
    from zen_garden.services.element_registry import ElementRegistry


class Element:
    """Class defining a standard Element."""

    # set label
    label: str = "set_elements"

    def __init__(
        self,
        element_name: str,
        config: Config,
        context: Context,
        energy_system: "EnergySystem",
        element_registry: "ElementRegistry",
        unit_handling: "UnitHandling",
    ):
        """Initialization of an element.

        :param element: element that is added to the model
        :param optimization_setup: The OptimizationSetup the element is part of
        """
        # set attributes
        self.name = element_name
        self._name = element_name
        # optimization setup
        self.config = config
        self.context = context
        # energy system
        self.energy_system = energy_system
        self.element_registry = element_registry
        self.unit_handling = unit_handling
        # set if aggregated
        self.aggregated = False
        # get input path
        self.input_path = self._get_input_path()
        # create DataInput object
        self.data_input = DataInput(
            element=self,
            energy_system=self.energy_system,
            unit_handling=self.unit_handling,
            config=self.config,
            context=self.context,
        )
        # dict to save the parameter units element-wise and to save them in the results
        self.units = {}

    def _get_input_path(self):
        """Get input path where input data is stored input_path."""
        # get technology type
        class_label = self.label
        # get path dictionary
        paths = self.context.paths
        # check if class is a subset
        if class_label not in paths.keys():
            subsets = self.config.analysis.subsets
            # iterate through subsets and check if class belongs to any of the subsets
            for set_name, subsets_list in subsets.items():
                if class_label in subsets_list:
                    class_label = set_name
                    break
        # get input path for current class_label
        return Path(paths[class_label][self.name]["folder"])

    def store_scenario_dict(self):
        """Stores scenario dict in each data input object."""
        # store scenario dict
        self.data_input.scenario_dict = self.context.scenario_dict

    ### --- classmethods to construct sets, parameters, variables, and constraints,
    #  corresponding to Element --- ###
    # Here, after defining EnergySystem-specific components,
    #   the components of the other classes are constructed
    @classmethod
    def construct_model_components(cls, optimization_setup):
        """Constructs the model components of the class <Element>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        raise NotImplementedError("TO BE REMOVED")

        logging.info("\n--- Construct model components ---\n")
        pid = os.getpid()

        # construct Sets
        t_start = time.perf_counter()
        cls.construct_sets(optimization_setup)
        t1 = time.perf_counter()
        if optimization_setup.solver.run_diagnostics:
            logging.info(f"Time to construct Sets: {t1 - t_start:0.1f} seconds")
            mem_usage = psutil.Process(pid).memory_info().rss / 1024**2
            logging.info(f"Memory usage: {mem_usage:0.1f} MB")

        # construct Params
        t0 = time.perf_counter()
        cls.construct_params(optimization_setup)
        t1 = time.perf_counter()
        if optimization_setup.solver.run_diagnostics:
            logging.info(f"Time to construct Params: {t1 - t0:0.1f} seconds")
            mem_usage = psutil.Process(pid).memory_info().rss / 1024**2
            logging.info(f"Memory usage: {mem_usage:0.1f} MB")

        # construct Vars
        t0 = time.perf_counter()
        cls.construct_vars(optimization_setup)
        t1 = time.perf_counter()
        if optimization_setup.solver.run_diagnostics:
            logging.info(f"Time to construct Vars: {t1 - t0:0.1f} seconds")
            mem_usage = psutil.Process(pid).memory_info().rss / 1024**2
            logging.info(f"Memory usage: {mem_usage:0.1f} MB")

        # construct Constraints
        t0 = time.perf_counter()
        cls.construct_constraints(optimization_setup)
        t1 = time.perf_counter()
        if optimization_setup.solver.run_diagnostics:
            logging.info(f"Time to construct Constraints: {t1 - t0:0.1f} seconds")
            mem_usage = psutil.Process(pid).memory_info().rss / 1024**2
            logging.info(f"Memory usage: {mem_usage:0.1f} MB")

        # construct Objective
        optimization_setup.energy_system.construct_objective()
        if optimization_setup.solver.run_diagnostics:
            logging.info(
                f"Total time to construct model components: "
                f"{time.perf_counter() - t_start:0.1f} seconds"
            )


class ElementConstructor(ABC):
    element_class: type[Element] = Element

    def __init__(
        self, config: Config, context: Context, element_registry: "ElementRegistry"
    ):
        self.config = config
        self.context = context
        self.element_registry = element_registry

    @abstractmethod
    def has_elements(self) -> bool:
        """Checks if the element has any elements to construct.

        :return: True if the element has elements, False otherwise
        """
        pass

    @abstractmethod
    def construct_sets(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the Sets of the class <Element>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        pass

    @abstractmethod
    def construct_params(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the Params of the class <Element>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        pass

    @abstractmethod
    def construct_vars(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the Vars of the class <Element>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        pass

    @abstractmethod
    def construct_constraints(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the Constraints of the class <Element>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        pass

    def create_custom_set(
        self, list_index: list[str], zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Creates custom set for model component.

        :param list_index: list of names of indices
        :param optimization_setup: The OptimizationSetup the element is part of
        :return: list_index: list of names of indices
        """
        list_index = list(list_index)  # make a copy of the list to avoid side effects
        sets = zen_model.sets

        # Case 1: all index sets are already defined in model and no set is indexed
        if all(
            index in sets.sets and not sets.is_indexed(index) for index in list_index
        ):
            list_sets = [sets[index] for index in list_index if index in sets]
            # return indices as cartesian product of sets
            custom_set: list[tuple[ZenSet, ...]] | list[ZenSet] = (
                list(itertools.product(*list_sets))
                if len(list_sets) > 1
                else list(list_sets[0])
            )
            return custom_set, list_index

        if list_index[0] not in energy_system.indexing_sets:
            raise NotImplementedError(
                f"Index <{list_index[0]}> is not in the indexing sets."
            )

        # Case 2: first index is indexed, build custom set based on first index
        custom_set = []
        for element in sets[list_index[0]]:
            append_element = True
            list_sets = []

            for index in list_index[1:]:
                # if the set already exist in model
                if index in sets:
                    append = self.handle_existing_set(index, element, sets, list_sets)
                    if not append:
                        raise NotImplementedError(
                            f"Index <{index}> is not known in sets."
                        )
                    continue

                # if index is set_location
                if index == "set_location":
                    self.handle_set_location_index(element, sets, list_sets)
                    continue

                # if set is built for pwa capex:
                if "set_capex" in index:
                    append_element = self.append_set_capex_index(element, sets, index)
                    continue

                # if set is used to determine if on-off behavior is modeled
                # exclude technologies which have no min_load
                if "on_off" in index:
                    append_element = self.append_on_off_modeled(element, index)
                    continue

                # split in capacity types of power and energy
                if index == "set_capacity_types":
                    self.handle_set_capacity_types_index(element, sets, list_sets)
                    continue

                raise NotImplementedError(f"Index <{index}> not known")

            # append indices to custom_set if element is supposed to be appended
            if append_element:
                if list_sets:
                    custom_set.extend(list(itertools.product([element], *list_sets)))
                else:
                    custom_set.extend([element])
        return custom_set, list_index

    def handle_existing_set(
        self, index: str, element: ZenSet, sets: IndexSet, list_sets: list[ZenSet]
    ):
        """Handles existing sets in the model.
        Returns True if handled, False if unknown.

        :param index: index to handle
        :param element: element to handle
        :param sets: sets of the optimization setup
        :param list_sets: list of sets to append
        """
        if not sets.is_indexed(index):
            list_sets.append(sets[index])
            return True
        elif sets.get_index_name(index) in sets.sets:
            list_sets.append(sets[index][element])
            return True
        return False

    def append_set_capex_index(self, element: str, sets: IndexSet, index: str) -> bool:
        """Checks if the capex of a technology needs to be modeled as pwa or linear.

        :param element: technology in model
        :param optimization_setup: The OptimizationSetup the element is part of
        :param index: index to check
        :return model_capex: Bool indicating if capex must be modeled as pwa or linear
        """
        if element not in sets["set_conversion_technologies"]:
            return False

        capex_is_pwa = self.element_registry.get_attribute_of_specific_element(
            self.element_class, element, "capex_is_pwa"
        )
        return not (
            ("linear" in index and capex_is_pwa)
            or ("pwa" in index and not capex_is_pwa)
        )

    def append_on_off_modeled(self, element: str, index: str) -> bool:
        """Checks if the on-off-behavior (min-load) of a technology needs to be modeled.

        :param element: technology in model
        :param optimization_setup: The OptimizationSetup the element is part of
        :param index: index to check
        :return model_on_off: Bool indicating if on-off-behavior needs to be modeled
        """
        model_on_off = self.check_on_off_modeled(element)
        return not (("set_no_on_off" in index and model_on_off) or (not model_on_off))

    def handle_set_location_index(
        self, element: str, sets: IndexSet, list_sets: list[ZenSet]
    ):
        """Handles the set_location index for the custom set.

        :param element: element to handle
        :param sets: sets of the optimization setup
        :param list_sets: list of sets to append
        """
        if (
            element in sets["set_conversion_technologies"]
            or element in sets["set_storage_technologies"]
            or element in sets["set_retrofitting_technologies"]
        ):
            list_sets.append(sets["set_nodes"])
        elif element in sets["set_transport_technologies"]:
            list_sets.append(sets["set_edges"])

    def handle_set_capacity_types_index(
        self, element: str, sets: IndexSet, list_sets: list[str]
    ):
        """Handles the set_capacity_types index for the custom set.

        :param element: element to handle
        :param sets: sets of the optimization setup
        :param optimization_setup: The OptimizationSetup the element is part of
        :param list_sets: list of sets to append
        """
        if element in sets["set_storage_technologies"]:
            list_sets.append(self.config.system.set_capacity_types)
        else:
            list_sets.append([self.config.system.set_capacity_types[0]])

    def check_on_off_modeled(self, tech: str):
        """Classmethod checks if on-off-behavior of a technology needs to be modeled.

        If the technology has a minimum load of 0 for all nodes and time steps, and all
        dependent carriers have a lower bound of 0 (only for conversion technologies
        modeled as pwa), then on-off-behavior is not necessary to model.

        :param tech: technology in model
        :param optimization_setup: The OptimizationSetup the element is part of
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
        zen_model: ZenModel,
        energy_system: "EnergySystem",
        name: str,
        doc: str,
        index_names: list[str],
        capacity_types: bool = False,
    ):
        """Adds a parameter to the optimization model for components without data.

        :param zen_model: The ZenModel the element is part of
        :param name: name of parameter
        :param doc: docstring of parameter
        """
        component_data, index_list, dict_of_units = self._initialize_component(
            zen_model, energy_system, name, index_names, capacity_types
        )
        component_data = self._ensure_pd_series_multi_index(component_data)
        data = component_data, index_list

        zen_model.parameters.add_parameter(
            name=name,
            doc=doc,
            data=data,
            dict_of_units=dict_of_units,
        )

    def _initialize_component(
        self,
        zen_model: ZenModel,
        energy_system: "EnergySystem",
        component_name: str,
        index_names: list[str],
        capacity_types: bool = False,
    ):
        """Initialize a modeling component by extracting the stored input data.

        Args:
            calling_class: class from where the method is called
            component_name: name of modeling component
            index_names: names of index sets, only if calling_class is not EnergySystem
            set_time_steps: time steps, only if calling_class is EnergySystem
            capacity_types: boolean if extracted for capacities
            component_data: data to initialize the component
        """

        if index_names is None:
            raise ValueError(f"Index names for {component_name} not specified")
        custom_set, index_list = self.create_custom_set(
            index_names, zen_model, energy_system
        )
        component_data, dict_of_units, attribute_is_series = (
            self.element_registry.get_attribute_of_all_elements(
                self.element_class,
                component_name,
                capacity_types=capacity_types,
                return_attribute_is_series=True,
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
                    f"the custom set {custom_set} cannot be used as a subindex of"
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
