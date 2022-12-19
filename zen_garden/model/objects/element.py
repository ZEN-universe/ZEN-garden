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
import cProfile, pstats
from zen_garden.preprocess.functions.extract_input_data import DataInput
from .energy_system import EnergySystem
from .component import Parameter,Variable,Constraint

class Element:
    # set label
    label = "set_elements"
    # empty list of elements
    list_of_elements = []

    def __init__(self,element):
        """ initialization of an element
        :param element: element that is added to the model"""
        # set attributes
        self.name = element
        # set if aggregated
        self.aggregated = False
        # get input path
        self.get_input_path()
        # create DataInput object
        self.datainput = DataInput(self,EnergySystem.get_system(),EnergySystem.get_analysis(),EnergySystem.get_solver(), EnergySystem.get_energy_system(),EnergySystem.get_unit_handling())
        # add element to list
        Element.add_element(self)

    def get_input_path(self):
        """ get input path where input data is stored input_path"""
        # get technology type
        class_label  = type(self)._get_class_label()
        # get path dictionary
        paths = EnergySystem.get_paths()
        # check if class is a subset
        if class_label not in paths.keys():
            subsets = EnergySystem.get_analysis()["subsets"]
            # iterate through subsets and check if class belongs to any of the subsets
            for set_name, subsets_list in subsets.items():
                if class_label in subsets_list:
                    class_label = set_name
                    break
        # get input path for current class_label
        self.input_path = paths[class_label][self.name]["folder"]

    def set_aggregated(self):
        """ this method sets self.aggregated to True """
        self.aggregated = True
    
    def is_aggregated(self):
        """ this method returns the aggregation status """
        return self.aggregated

    def overwrite_time_steps(self,base_time_steps):
        """ overwrites time steps. Must be implemented in child classes """
        raise NotImplementedError("overwrite_time_steps must be implemented in child classes!")

    ### --- classmethods --- ###
    # setter/getter classmethods
    @classmethod
    def add_element(cls,element):
        """ add element to element list. Inherited by child classes.
        :param element: new element that is to be added to the list """
        cls.list_of_elements.append(element)

    @classmethod
    def get_all_elements(cls):
        """ get all elements in class. Inherited by child classes.
        :return cls.list_of_elements: list of elements in this class """
        return cls.list_of_elements

    @classmethod
    def get_all_names_of_elements(cls):
        """ get all names of elements in class. Inherited by child classes.
        :return names_of_elements: list of names of elements in this class """
        _elements_in_class = cls.get_all_elements()
        names_of_elements = []
        for _element in _elements_in_class:
            names_of_elements.append(_element.name)
        return names_of_elements
        
    @classmethod
    def get_element(cls,name:str):
        """ get single element in class by name. Inherited by child classes.
        :param name: name of element
        :return element: return element whose name is matched """
        for _element in cls.list_of_elements:
            if _element.name == name:
                return _element
        return None

    @classmethod
    def get_all_subclasses(cls):
        """ get all subclasses (child classes) of cls 
        :return subclasses: subclasses of cls """
        return cls.__subclasses__()

    @classmethod
    def get_attribute_of_all_elements(cls,attribute_name:str,capacity_types = False,return_attribute_is_series = False):
        """ get attribute values of all elements in this class 
        :param attribute_name: str name of attribute
        :param capacity_types: boolean if attributes extracted for all capacity types
        :param return_attribute_is_series: boolean if information on attribute type is returned
        :return dict_of_attributes: returns dict of attribute values
        :return attribute_is_series: return information on attribute type """
        system            = EnergySystem.get_system()
        _class_elements    = cls.get_all_elements()
        dict_of_attributes  = {}
        attribute_is_series = False
        for _element in _class_elements:
            if not capacity_types:
                dict_of_attributes,attribute_is_series = cls.append_attribute_of_element_to_dict(_element,attribute_name,dict_of_attributes)
            # if extracted for both capacity types
            else:
                for capacity_type in system["set_capacity_types"]:
                    # append energy only for storage technologies
                    if capacity_type == system["set_capacity_types"][0] or _element.name in system["setStorageTechnologies"]:
                        dict_of_attributes,attribute_is_series = cls.append_attribute_of_element_to_dict(_element, attribute_name, dict_of_attributes,capacity_type)
        if return_attribute_is_series:
            return dict_of_attributes,attribute_is_series
        else:
            return dict_of_attributes

    @classmethod
    def append_attribute_of_element_to_dict(cls,_element,attribute_name,dict_of_attributes,capacity_type = None):
        """ get attribute values of all elements in this class
        :param _element: element of class
        :param attribute_name: str name of attribute
        :param dict_of_attributes: dict of attribute values
        :param capacity_type: capacity type for which attribute extracted. If None, not listed in key
        :return dict_of_attributes: returns dict of attribute values """
        attribute_is_series = False
        system = EnergySystem.get_system()
        # add Energy for energy capacity type
        if capacity_type == system["set_capacity_types"][1]:
            attribute_name += "_energy"
        assert hasattr(_element, attribute_name), f"Element {_element.name} does not have attribute {attribute_name}"
        _attribute = getattr(_element, attribute_name)
        assert not isinstance(_attribute, pd.DataFrame), f"Not yet implemented for pd.DataFrames. Wrong format for element {_element.name}"
        # add attribute to dict_of_attributes
        if isinstance(_attribute, dict):
            dict_of_attributes.update({(_element.name,)+(key,):val for key,val in _attribute.items()})
        elif isinstance(_attribute, pd.Series) and "pwa" not in attribute_name:
            if capacity_type:
                _combined_key = (_element.name,capacity_type)
            else:
                _combined_key = _element.name
            if len(_attribute) > 1:
                dict_of_attributes[_combined_key] = _attribute
                attribute_is_series = True
            else:
                dict_of_attributes[_combined_key] = _attribute.squeeze()
                attribute_is_series = False
        elif isinstance(_attribute, int):
            if capacity_type:
                dict_of_attributes[(_element.name,capacity_type)] = [_attribute]
            else:
                dict_of_attributes[_element.name] = [_attribute]
        else:
            if capacity_type:
                dict_of_attributes[(_element.name,capacity_type)] = _attribute
            else:
                dict_of_attributes[_element.name] = _attribute
        return dict_of_attributes, attribute_is_series

    @classmethod
    def get_attribute_of_specific_element(cls,element_name:str,attribute_name:str):
        """ get attribute of specific element in class
        :param element_name: str name of element
        :param attribute_name: str name of attribute
        :return attribute_value: value of attribute"""
        # get element
        _element = cls.get_element(element_name)
        # assert that _element exists and has attribute
        assert _element, f"Element {element_name} not in class {cls}"
        assert hasattr(_element,attribute_name),f"Element {element_name} does not have attribute {attribute_name}"
        attribute_value = getattr(_element,attribute_name)
        return attribute_value

    @classmethod
    def _get_class_label(cls):
        """ returns label of class """
        return cls.label

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Element --- ###
    # Here, after defining EnergySystem-specific components, the components of the other classes are constructed
    @classmethod
    def construct_model_components(cls):
        """ constructs the model components of the class <Element> """
        logging.info("\n--- Construct model components ---\n")
        # construct pe.Sets
        cls.construct_sets()
        # construct pe.Params
        cls.construct_params()
        # construct pe.Vars
        cls.construct_vars()
        # construct pe.Constraints
        cls.construct_constraints()
        # construct pe.Objective
        EnergySystem.construct_objective()

    @classmethod
    def construct_sets(cls):
        """ constructs the pe.Sets of the class <Element> """
        logging.info("Construct pe.Sets")
        # construct pe.Sets of energy system
        EnergySystem.construct_sets()
        # construct pe.Sets of class elements
        model = EnergySystem.get_pyomo_model()
        # operational time steps
        model.setTimeStepsOperation = pe.Set(
            model.set_elements,
            initialize=cls.get_attribute_of_all_elements("setTimeStepsOperation"),
            doc="Set of time steps in operation for all technologies. Dimensions: set_elements"
        )
        # construct pe.Sets of the child classes
        for subclass in cls.get_all_subclasses():
            print(subclass.__name__)
            subclass.construct_sets()

    @classmethod
    def construct_params(cls):
        """ constructs the pe.Params of the class <Element> """
        logging.info("Construct pe.Params")
        # initialize parameterObject
        Parameter()
        # construct pe.Params of energy system
        EnergySystem.construct_params()
        # construct pe.Sets of class elements
        # operational time step duration
        Parameter.add_parameter(
            name="time_steps_operation_duration",
            data= EnergySystem.initialize_component(cls,"time_steps_operation_duration",index_names=["set_elements","setTimeStepsOperation"]),#.astype(int),
            # doc="Parameter which specifies the time step duration in operation for all technologies. Dimensions: set_elements, setTimeStepsOperation"
            doc="Parameter which specifies the time step duration in operation for all technologies"
        )
        # construct pe.Params of the child classes
        for subclass in cls.get_all_subclasses():
            subclass.construct_params()

    @classmethod
    def construct_vars(cls):
        """ constructs the pe.Vars of the class <Element> """
        logging.info("Construct pe.Vars")
        # initialize variableObject
        Variable()
        # construct pe.Vars of energy system
        EnergySystem.construct_vars()
        # construct pe.Vars of the child classes
        for subclass in cls.get_all_subclasses():
            subclass.construct_vars()

    @classmethod
    def construct_constraints(cls):
        """ constructs the pe.Constraints of the class <Element> """
        logging.info("Construct pe.Constraints")
        # initialize constraintObject
        Constraint()
        # construct pe.Constraints of energy system
        EnergySystem.construct_constraints()
        # construct pe.Constraints of the child classes
        for subclass in cls.get_all_subclasses():
            subclass.construct_constraints()

    @classmethod
    def create_custom_set(cls,list_index):
        """ creates custom set for model component 
        :param list_index: list of names of indices
        :return custom_set: custom set index
        :return list_index: list of names of indices """
        list_index_overwrite = copy.copy(list_index)
        model = EnergySystem.get_pyomo_model()
        indexing_sets = EnergySystem.get_indexing_sets()
        # check if all index sets are already defined in model and no set is indexed
        if all([(hasattr(model,index) and not model.find_component(index).is_indexed()) for index in list_index]):
            # check if no set is indexed
            list_sets = []
            # iterate through indices
            for index in list_index:
                # if the set already exists in model
                if hasattr(model,index):
                    # append set to list
                    list_sets.append(model.find_component(index))
            # return indices as cartesian product of sets
            if len(list_index) > 1:
                custom_set = list(itertools.product(*list_sets))
            else:
                custom_set = list(list_sets[0])
            return custom_set,list_index
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
                        if hasattr(model,index):
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
                            # if element in setConversionTechnologies or setStorageTechnologies, append setNodes
                            if element in model.setConversionTechnologies or element in model.setStorageTechnologies:
                                list_sets.append(model.setNodes)
                            # if element in setTransportTechnologies
                            elif element in model.setTransportTechnologies:
                                list_sets.append(model.setEdges)
                        # if set is built for pwa capex:
                        elif "set_capex" in index:
                            if element in model.setConversionTechnologies:
                                _capex_is_pwa = cls.get_attribute_of_specific_element(element,"capexIsPWA")
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
                        # if set is built for pwa converEfficiency:
                        elif "set_conver_efficiency" in index:
                            if element in model.setConversionTechnologies: # or element in model.setStorageTechnologies:
                                _conver_efficiency_is_pwa = cls.get_attribute_of_specific_element(element, "converEfficiencyIsPWA")
                                dependent_carrier = list(model.setDependentCarriers[element])
                                # TODO for more than one carrier
                                # _PWAConverEfficiency = cls.get_attribute_of_specific_element(element,"PWAConverEfficiency")
                                # dependent_carrier_pwa     = _PWAConverEfficiency["pwa_variables"]
                                if "linear" in index and not _conver_efficiency_is_pwa:
                                    list_sets.append(dependent_carrier)
                                elif "pwa" in index and _conver_efficiency_is_pwa:
                                    list_sets.append(dependent_carrier)
                                else:
                                    list_sets.append([])
                                list_index_overwrite = list(map(lambda x: x.replace(index, 'setCarriers'), list_index))
                            # Transport or Storage technology
                            else:
                                append_element = False
                                break
                        # if set is used to determine if on-off behavior is modeled
                        # exclude technologies which have no min_load and dependentCarrierFlow at referenceCarrierFlow = 0 is also equal to 0
                        elif "on_off" in index:
                            model_on_off = cls.check_on_off_modeled(element)
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
                            system = EnergySystem.get_system()
                            if element in model.setStorageTechnologies:
                                list_sets.append(system["set_capacity_types"])
                            else:
                                list_sets.append([system["set_capacity_types"][0]])
                        else:
                            raise NotImplementedError(f"Index <{index}> not known")
                    # append indices to custom_set if element is supposed to be appended
                    if append_element:
                        if list_sets:
                            custom_set.extend(list(itertools.product([element],*list_sets)))
                        else:
                            custom_set.extend([element])
                return custom_set,list_index_overwrite
            else:
                raise NotImplementedError

    @classmethod
    def check_on_off_modeled(cls,tech):
        """ this classmethod checks if the on-off-behavior of a technology needs to be modeled.
        If the technology has a minimum load of 0 for all nodes and time steps,
        and all dependent carriers have a lower bound of 0 (only for conversion technologies modeled as pwa),
        then on-off-behavior is not necessary to model
        :param tech: technology in model
        :returns model_on_off: boolean to indicate that on-off-behavior modeled """
        model = EnergySystem.get_pyomo_model()

        model_on_off = True
        # check if any min
        _unique_min_load = list(set(cls.get_attribute_of_specific_element(tech,"min_load").values))
        # if only one unique min_load which is zero
        if len(_unique_min_load) == 1 and _unique_min_load[0] == 0:
            # if not a conversion technology, break for current technology
            if tech not in model.setConversionTechnologies:
                model_on_off = False
            # if a conversion technology, check if all dependentCarrierFlow at referenceCarrierFlow = 0 equal to 0
            else:
                # if technology is approximated (by either pwa or linear)
                _isPWA = cls.get_attribute_of_specific_element(tech,"converEfficiencyIsPWA")
                # if not modeled as pwa
                if not _isPWA:
                    model_on_off = False
                else:
                    _PWAParameter = cls.get_attribute_of_specific_element(tech,"PWAConverEfficiency")
                    # iterate through all dependent carriers and check if all lower bounds are equal to 0
                    _only_zero_dependent_bound = True
                    for PWAVariable in _PWAParameter["pwa_variables"]:
                        if _PWAParameter["bounds"][PWAVariable][0] != 0:
                            _only_zero_dependent_bound = False
                    if _only_zero_dependent_bound:
                        model_on_off = False
        # return
        return model_on_off
