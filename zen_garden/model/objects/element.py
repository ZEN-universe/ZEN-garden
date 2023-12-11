"""
:Title:          ZEN-GARDEN
:Created:        October-2021
:Authors:        Alissa Ganter (aganter@ethz.ch),
                Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Class defining a standard Element. Contains methods to add parameters, variables and constraints to the
optimization problem. Parent class of the Carrier and Technology classes .The class takes the concrete
optimization model as an input.
"""
import copy
import itertools
import logging
import os
import psutil
import time
from pathlib import Path
from zen_garden.preprocess.extract_input_data import DataInput

class Element:
    """
    Class defining a standard Element
    """
    # set label
    label = "set_elements"

    def __init__(self, element: str, optimization_setup):
        """ initialization of an element

        :param element: element that is added to the model
        :param optimization_setup: The OptimizationSetup the element is part of """
        # set attributes
        self.name = element
        self._name = element
        # optimization setup
        self.optimization_setup = optimization_setup
        # energy system
        self.energy_system = optimization_setup.energy_system
        # set if aggregated
        self.aggregated = False
        # get input path
        self.get_input_path()
        # create DataInput object
        self.data_input = DataInput(element=self, system=self.optimization_setup.system,
                                    analysis=self.optimization_setup.analysis, solver=self.optimization_setup.solver,
                                    energy_system=self.energy_system, unit_handling=self.energy_system.unit_handling)

    def get_input_path(self):
        """ get input path where input data is stored input_path"""
        # get technology type
        class_label = self.label
        # get path dictionary
        paths = self.optimization_setup.paths
        # check if class is a subset
        if class_label not in paths.keys():
            subsets = self.optimization_setup.analysis["subsets"]
            # iterate through subsets and check if class belongs to any of the subsets
            for set_name, subsets_list in subsets.items():
                if class_label in subsets_list:
                    class_label = set_name
                    break
        # get input path for current class_label
        self.input_path = Path(paths[class_label][self.name]["folder"])

    def store_scenario_dict(self):
        """ stores scenario dict in each data input object """
        # store scenario dict
        self.data_input.scenario_dict = self.optimization_setup.scenario_dict

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Element --- ###
    # Here, after defining EnergySystem-specific components, the components of the other classes are constructed
    @classmethod
    def construct_model_components(cls, optimization_setup):
        """ constructs the model components of the class <Element>

        :param optimization_setup: The OptimizationSetup the element is part of """
        logging.info("\n--- Construct model components ---\n")
        pid = os.getpid()
        # construct pe.Sets
        t_start = time.perf_counter()
        cls.construct_sets(optimization_setup)
        t1 = time.perf_counter()
        logging.info(f"Time to construct pe.Sets: {t1 - t_start:0.4f} seconds")
        logging.info(f"Memory usage: {psutil.Process(pid).memory_info().rss / 1024 ** 2} MB")
        # construct pe.Params
        t0 = time.perf_counter()
        cls.construct_params(optimization_setup)
        t1 = time.perf_counter()
        logging.info(f"Time to construct pe.Params: {t1 - t0:0.4f} seconds")
        logging.info(f"Memory usage: {psutil.Process(pid).memory_info().rss / 1024 ** 2} MB")
        # construct pe.Vars
        t0 = time.perf_counter()
        cls.construct_vars(optimization_setup)
        t1 = time.perf_counter()
        logging.info(f"Time to construct pe.Vars: {t1 - t0:0.4f} seconds")
        logging.info(f"Memory usage: {psutil.Process(pid).memory_info().rss / 1024 ** 2} MB")
        # construct pe.Constraints
        t0 = time.perf_counter()
        cls.construct_constraints(optimization_setup)
        t1 = time.perf_counter()
        logging.info(f"Time to construct pe.Constraints: {t1 - t0:0.4f} seconds")
        logging.info(f"Memory usage: {psutil.Process(pid).memory_info().rss / 1024 ** 2} MB")
        # construct pe.Objective
        optimization_setup.energy_system.construct_objective()
        t_end = time.perf_counter()
        logging.info(f"Total time to construct model components: {t_end - t_start:0.4f} seconds")

    @classmethod
    def construct_sets(cls, optimization_setup):
        """ constructs the pe.Sets of the class <Element>

        :param optimization_setup: The OptimizationSetup the element is part of """
        logging.info("Construct pe.Sets")
        # construct pe.Sets of energy system
        optimization_setup.energy_system.construct_sets()
        # construct pe.Sets of the child classes
        for subclass in cls.__subclasses__():
            subclass.construct_sets(optimization_setup)

    @classmethod
    def construct_params(cls, optimization_setup):
        """ constructs the pe.Params of the class <Element>

        :param optimization_setup: The OptimizationSetup the element is part of """
        logging.info("Construct pe.Params")
        # construct pe.Params of energy system
        optimization_setup.energy_system.construct_params()
        # construct pe.Params of the child classes
        for subclass in cls.__subclasses__():
            subclass.construct_params(optimization_setup)

    @classmethod
    def construct_vars(cls, optimization_setup):
        """ constructs the pe.Vars of the class <Element>

        :param optimization_setup: The OptimizationSetup the element is part of """
        logging.info("Construct pe.Vars")
        # construct pe.Vars of energy system
        optimization_setup.energy_system.construct_vars()
        # construct pe.Vars of the child classes
        for subclass in cls.__subclasses__():
            subclass.construct_vars(optimization_setup)

    @classmethod
    def construct_constraints(cls, optimization_setup):
        """ constructs the pe.Constraints of the class <Element>

        :param optimization_setup: The OptimizationSetup the element is part of """
        logging.info("Construct pe.Constraints")
        # construct pe.Constraints of energy system
        optimization_setup.energy_system.construct_constraints()
        # construct pe.Constraints of the child classes
        for subclass in cls.__subclasses__():
            logging.info(f"Construct pe.Constraints of {subclass.__name__}")
            subclass.construct_constraints(optimization_setup)

    @classmethod
    def create_custom_set(cls, list_index, optimization_setup):
        """ creates custom set for model component 

        :param list_index: list of names of indices
        :param optimization_setup: The OptimizationSetup the element is part of
        :return list_index: list of names of indices """
        list_index_overwrite = copy.copy(list_index)
        sets = optimization_setup.sets
        indexing_sets = optimization_setup.energy_system.indexing_sets
        # check if all index sets are already defined in model and no set is indexed
        if all([(index in sets.sets and not sets.is_indexed(index)) for index in list_index]):
            # check if no set is indexed
            list_sets = []
            # iterate through indices
            for index in list_index:
                # if the set already exists in model
                if index in sets:
                    # append set to list
                    list_sets.append(sets[index])
            # return indices as cartesian product of sets
            if len(list_index) > 1:
                custom_set = list(itertools.product(*list_sets))
            else:
                custom_set = list(list_sets[0])
            return custom_set, list_index
        # at least one set is not yet defined
        else:
            # ugly, but if first set is indexing_set
            if list_index[0] in indexing_sets:
                # empty custom index set
                custom_set = []
                # iterate through
                for element in sets[list_index[0]]:
                    # default: append element
                    append_element = True
                    # create empty list of sets
                    list_sets = []
                    # iterate through indices without first index
                    for index in list_index[1:]:
                        # if the set already exist in model
                        if index in sets:
                            # if not indexed
                            if not sets.is_indexed(index):
                                list_sets.append(sets[index])
                            # if indexed by first entry
                            elif sets.get_index_name(index) in indexing_sets:
                                list_sets.append(sets[index][element])
                            else:
                                raise NotImplementedError
                        # if index is set_location
                        elif index == "set_location":
                            # if element in set_conversion_technologies or set_storage_technologies, append set_nodes
                            if (element in sets["set_conversion_technologies"] or element in sets["set_storage_technologies"] \
                                    or element in sets["set_retrofitting_technologies"]):
                                list_sets.append(sets["set_nodes"])
                            # if element in set_transport_technologies
                            elif element in sets["set_transport_technologies"]:
                                list_sets.append(sets["set_edges"])
                        # if set is built for pwa capex:
                        elif "set_capex" in index:
                            if element in sets["set_conversion_technologies"]:
                                capex_is_pwa = optimization_setup.get_attribute_of_specific_element(cls, element, "capex_is_pwa")
                                # if technology is modeled as pwa, break for linear index
                                if "linear" in index and capex_is_pwa:
                                    append_element = False
                                    break
                                # if technology is not modeled as pwa, break for pwa index
                                elif "pwa" in index and not capex_is_pwa:
                                    append_element = False
                                    break
                            # Transport or Storage technology
                            else:
                                append_element = False
                                break
                        # if set is used to determine if on-off behavior is modeled
                        # exclude technologies which have no min_load
                        elif "on_off" in index:
                            model_on_off = cls.check_on_off_modeled(element, optimization_setup)
                            if "set_no_on_off" in index:
                                # if modeled as on off, do not append to set_no_on_off
                                if model_on_off:
                                    append_element = False
                                    break
                            else:
                                # if not modeled as on off, do not append to set_on_off
                                if not model_on_off:
                                    append_element = False
                                    break
                        # split in capacity types of power and energy
                        elif index == "set_capacity_types":
                            system = optimization_setup.system
                            if element in sets["set_storage_technologies"]:
                                list_sets.append(system["set_capacity_types"])
                            else:
                                list_sets.append([system["set_capacity_types"][0]])
                        else:
                            raise NotImplementedError(f"Index <{index}> not known")
                    # append indices to custom_set if element is supposed to be appended
                    if append_element:
                        if list_sets:
                            custom_set.extend(list(itertools.product([element], *list_sets)))
                        else:
                            custom_set.extend([element])
                return custom_set, list_index_overwrite
            else:
                raise NotImplementedError

    @classmethod
    def check_on_off_modeled(cls, tech, optimization_setup):
        """ this classmethod checks if the on-off-behavior of a technology needs to be modeled.
        If the technology has a minimum load of 0 for all nodes and time steps,
        and all dependent carriers have a lower bound of 0 (only for conversion technologies modeled as pwa),
        then on-off-behavior is not necessary to model

        :param tech: technology in model
        :param optimization_setup: The OptimizationSetup the element is part of
        :return model_on_off: Bool indicating if on-off-behaviour (min load) needs to be modeled"""
        # check if any min load
        unique_min_load = list(set(optimization_setup.get_attribute_of_specific_element(cls, tech, "min_load").values))
        # if only one unique min_load which is zero
        if len(unique_min_load) == 1 and unique_min_load[0] == 0:
            model_on_off = False
        # otherwise modeled as on-off
        else:
            model_on_off = True
        # return
        return model_on_off


class GenericRule(object):
    """
    This class implements a generic rule for the model, which can be used to init the other rules of the technologies
    and carriers
    """

    def __init__(self, optimization_setup):
        """
        Constructor for generic rule
        :param optimization_setup: The optimization setup to use for the setup
        """

        self.optimization_setup = optimization_setup
        self.system = self.optimization_setup.system
        self.analysis = self.optimization_setup.analysis
        self.sets = self.optimization_setup.sets
        self.model = self.optimization_setup.model
        self.parameters = self.optimization_setup.parameters
        self.variables = self.model.variables
        self.constraints = self.optimization_setup.constraints
        self.energy_system = self.optimization_setup.energy_system
        self.time_steps = self.energy_system.time_steps
