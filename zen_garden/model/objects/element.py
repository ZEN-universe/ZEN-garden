"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class defining a standard Element. Contains methods to add parameters, variables and constraints to the
                optimization problem. Parent class of the Carrier and Technology classes .The class takes the concrete
                optimization model as an input.
==========================================================================================================================================================================="""
import copy
import itertools
import logging

import pandas as pd
import pyomo.environ as pe

from zen_garden.preprocess.functions.extract_input_data import DataInput
from .component import Parameter, Variable, Constraint
from .energy_system import EnergySystem


class Element:
    # set label
    label = "set_elements"

    def __init__(self, element: str, energy_system: EnergySystem):
        """ initialization of an element
        :param element: element that is added to the model
        :param energy_system: The energy system the element is part of """
        # set attributes
        self.name = element
        # energy system
        self.energy_system = energy_system
        # set if aggregated
        self.aggregated = False
        # get input path
        self.get_input_path()
        # create DataInput object
        self.data_input = DataInput(element=self, system=self.energy_system.system, analysis=self.energy_system.analysis,
                                    solver=self.energy_system.solver, energy_system=self.energy_system,
                                    unit_handling=self.energy_system.unit_handling)

    def get_input_path(self):
        """ get input path where input data is stored input_path"""
        # get technology type
        class_label = self.label
        # get path dictionary
        paths = self.energy_system.paths
        # check if class is a subset
        if class_label not in paths.keys():
            subsets = self.energy_system.analysis["subsets"]
            # iterate through subsets and check if class belongs to any of the subsets
            for set_name, subsets_list in subsets.items():
                if class_label in subsets_list:
                    class_label = set_name
                    break
        # get input path for current class_label
        self.input_path = paths[class_label][self.name]["folder"]

    def overwrite_time_steps(self, base_time_steps):
        """ overwrites time steps. Must be implemented in child classes """
        raise NotImplementedError("overwrite_time_steps must be implemented in child classes!")

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Element --- ###
    # Here, after defining EnergySystem-specific components, the components of the other classes are constructed
    @classmethod
    def construct_model_components(cls, energy_system: EnergySystem):
        """ constructs the model components of the class <Element>
        :param energy_system: The Energy system to add everything"""
        logging.info("\n--- Construct model components ---\n")
        # construct pe.Sets
        cls.construct_sets(energy_system)
        # construct pe.Params
        cls.construct_params(energy_system)
        # construct pe.Vars
        cls.construct_vars(energy_system)
        # construct pe.Constraints
        cls.construct_constraints(energy_system)
        # construct pe.Objective
        energy_system.construct_objective()

    @classmethod
    def construct_sets(cls, energy_system: EnergySystem):
        """ constructs the pe.Sets of the class <Element>
        :param energy_system: The Energy system to add everything"""
        logging.info("Construct pe.Sets")
        # construct pe.Sets of energy system
        energy_system.construct_sets()
        # construct pe.Sets of class elements
        model = energy_system.pyomo_model
        # operational time steps
        model.set_time_steps_operation = pe.Set(model.set_elements, initialize=energy_system.get_attribute_of_all_elements(cls, "set_time_steps_operation"),
            doc="Set of time steps in operation for all technologies. Dimensions: set_elements")
        # construct pe.Sets of the child classes
        for subclass in cls.__subclasses__():
            print(subclass.__name__)
            subclass.construct_sets(energy_system)

    @classmethod
    def construct_params(cls, energy_system: EnergySystem):
        """ constructs the pe.Params of the class <Element>
        :param energy_system: The Energy system to add everything"""
        logging.info("Construct pe.Params")
        # construct pe.Params of energy system
        energy_system.construct_params()
        # construct pe.Sets of class elements
        # operational time step duration
        energy_system.parameters.add_parameter(name="time_steps_operation_duration",
            data=energy_system.initialize_component(cls, "time_steps_operation_duration", index_names=["set_elements", "set_time_steps_operation"]),  # .astype(int),
            # doc="Parameter which specifies the time step duration in operation for all technologies. Dimensions: set_elements, set_time_steps_operation"
            doc="Parameter which specifies the time step duration in operation for all technologies")
        # construct pe.Params of the child classes
        for subclass in cls.__subclasses__():
            subclass.construct_params(energy_system)

    @classmethod
    def construct_vars(cls, energy_system: EnergySystem):
        """ constructs the pe.Vars of the class <Element>
        :param energy_system: The Energy system to add everything"""
        logging.info("Construct pe.Vars")
        # construct pe.Vars of energy system
        energy_system.construct_vars()
        # construct pe.Vars of the child classes
        for subclass in cls.__subclasses__():
            subclass.construct_vars(energy_system)

    @classmethod
    def construct_constraints(cls, energy_system: EnergySystem):
        """ constructs the pe.Constraints of the class <Element>
        :param energy_system: The Energy system to add everything"""
        logging.info("Construct pe.Constraints")
        # construct pe.Constraints of energy system
        energy_system.construct_constraints()
        # construct pe.Constraints of the child classes
        for subclass in cls.__subclasses__():
            subclass.construct_constraints()

    @classmethod
    def create_custom_set(cls, list_index, energy_system: EnergySystem):
        """ creates custom set for model component 
        :param list_index: list of names of indices
        :param energy_system: The Energy system of the elements
        :return list_index: list of names of indices """
        list_index_overwrite = copy.copy(list_index)
        model = energy_system.pyomo_model
        indexing_sets = energy_system.indexing_sets
        # check if all index sets are already defined in model and no set is indexed
        if all([(hasattr(model, index) and not model.find_component(index).is_indexed()) for index in list_index]):
            # check if no set is indexed
            list_sets = []
            # iterate through indices
            for index in list_index:
                # if the set already exists in model
                if hasattr(model, index):
                    # append set to list
                    list_sets.append(model.find_component(index))
            # return indices as cartesian product of sets
            if len(list_index) > 1:
                custom_set = list(itertools.product(*list_sets))
            else:
                custom_set = list(list_sets[0])
            return custom_set, list_index
        # at least one set is not yet defined
        else:
            # ugly, but if first set is indexingSet
            if list_index[0] in indexing_sets:
                # empty custom index set
                custom_set = []
                # iterate through
                for element in model.find_component(list_index[0]):
                    # default: append element
                    append_element = True
                    # create empty list of sets
                    list_sets = []
                    # iterate through indices without first index
                    for index in list_index[1:]:
                        # if the set already exist in model
                        if hasattr(model, index):
                            # if not indexed
                            if not model.find_component(index).is_indexed():
                                list_sets.append(model.find_component(index))
                            # if indexed by first entry
                            elif model.find_component(index).index_set().name in indexing_sets:
                                list_sets.append(model.find_component(index)[element])
                            else:
                                raise NotImplementedError
                        # if index is set_location
                        elif index == "set_location":
                            # if element in set_conversion_technologies or set_storage_technologies, append set_nodes
                            if element in model.set_conversion_technologies or element in model.set_storage_technologies:
                                list_sets.append(model.set_nodes)
                            # if element in set_transport_technologies
                            elif element in model.set_transport_technologies:
                                list_sets.append(model.set_edges)
                        # if set is built for pwa capex:
                        elif "set_capex" in index:
                            if element in model.set_conversion_technologies:
                                _capex_is_pwa = energy_system.get_attribute_of_specific_element(cls, element, "capex_is_pwa")
                                # if technology is modeled as pwa, break for linear index
                                if "linear" in index and _capex_is_pwa:
                                    append_element = False
                                    break
                                # if technology is not modeled as pwa, break for pwa index
                                elif "pwa" in index and not _capex_is_pwa:
                                    append_element = False
                                    break
                            # Transport or Storage technology
                            else:
                                append_element = False
                                break
                        # if set is built for pwa conver_efficiency:
                        elif "set_conver_efficiency" in index:
                            if element in model.set_conversion_technologies:  # or element in model.set_storage_technologies:
                                _conver_efficiency_is_pwa = energy_system.get_attribute_of_specific_element(cls, element, "conver_efficiency_is_pwa")
                                dependent_carrier = list(model.set_dependent_carriers[element])
                                # TODO for more than one carrier
                                # _pwa_conver_efficiency = cls.get_attribute_of_specific_element(element,"pwa_conver_efficiency")
                                # dependent_carrier_pwa     = _pwa_conver_efficiency["pwa_variables"]
                                if "linear" in index and not _conver_efficiency_is_pwa:
                                    list_sets.append(dependent_carrier)
                                elif "pwa" in index and _conver_efficiency_is_pwa:
                                    list_sets.append(dependent_carrier)
                                else:
                                    list_sets.append([])
                                list_index_overwrite = list(map(lambda x: x.replace(index, 'set_carriers'), list_index))
                            # Transport or Storage technology
                            else:
                                append_element = False
                                break
                        # if set is used to determine if on-off behavior is modeled
                        # exclude technologies which have no min_load and dependentCarrierFlow at reference_carrierFlow = 0 is also equal to 0
                        elif "on_off" in index:
                            model_on_off = cls.check_on_off_modeled(element, energy_system)
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
                            system = energy_system.system
                            if element in model.set_storage_technologies:
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
    def check_on_off_modeled(cls, tech, energy_system):
        """ this classmethod checks if the on-off-behavior of a technology needs to be modeled.
        If the technology has a minimum load of 0 for all nodes and time steps,
        and all dependent carriers have a lower bound of 0 (only for conversion technologies modeled as pwa),
        then on-off-behavior is not necessary to model
        :param tech: technology in model
        :param energy_system: The Energy System of the element
        :returns model_on_off: boolean to indicate that on-off-behavior modeled """
        model = energy_system.pyomo_model

        model_on_off = True
        # check if any min
        _unique_min_load = list(set(energy_system.get_attribute_of_specific_element(cls, tech, "min_load").values))
        # if only one unique min_load which is zero
        if len(_unique_min_load) == 1 and _unique_min_load[0] == 0:
            # if not a conversion technology, break for current technology
            if tech not in model.set_conversion_technologies:
                model_on_off = False
            # if a conversion technology, check if all dependentCarrierFlow at reference_carrierFlow = 0 equal to 0
            else:
                # if technology is approximated (by either pwa or linear)
                _isPWA = energy_system.get_attribute_of_specific_element(cls, tech, "conver_efficiency_is_pwa")
                # if not modeled as pwa
                if not _isPWA:
                    model_on_off = False
                else:
                    _pwa_parameter = energy_system.get_attribute_of_specific_element(cls, tech, "pwa_conver_efficiency")
                    # iterate through all dependent carriers and check if all lower bounds are equal to 0
                    _only_zero_dependent_bound = True
                    for PWAVariable in _pwa_parameter["pwa_variables"]:
                        if _pwa_parameter["bounds"][PWAVariable][0] != 0:
                            _only_zero_dependent_bound = False
                    if _only_zero_dependent_bound:
                        model_on_off = False
        # return
        return model_on_off
