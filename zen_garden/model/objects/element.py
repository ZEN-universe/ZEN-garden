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
    listOfElements = []

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
        Element.addElement(self)

    def get_input_path(self):
        """ get input path where input data is stored input_path"""
        # get system information
        system      = EnergySystem.get_system()
        # get technology type
        classLabel  = type(self).getClassLabel()
        # get path dictionary
        paths = EnergySystem.get_paths()
        # check if class is a subset
        if classLabel not in paths.keys():
            subsets = EnergySystem.get_analysis()["subsets"]
            # iterate through subsets and check if class belongs to any of the subsets
            for setName, subsetsList in subsets.items():
                if classLabel in subsetsList:
                    classLabel = setName
                    break
        # get input path for current classLabel
        self.input_path = paths[classLabel][self.name]["folder"]

    def setAggregated(self):
        """ this method sets self.aggregated to True """
        self.aggregated = True
    
    def isAggregated(self):
        """ this method returns the aggregation status """
        return self.aggregated

    def overwrite_time_steps(self,base_time_steps):
        """ overwrites time steps. Must be implemented in child classes """
        raise NotImplementedError("overwrite_time_steps must be implemented in child classes!")

    ### --- classmethods --- ###
    # setter/getter classmethods
    @classmethod
    def addElement(cls,element):
        """ add element to element list. Inherited by child classes.
        :param element: new element that is to be added to the list """
        cls.listOfElements.append(element)

    @classmethod
    def get_all_elements(cls):
        """ get all elements in class. Inherited by child classes.
        :return cls.listOfElements: list of elements in this class """
        return cls.listOfElements

    @classmethod
    def getAllNamesOfElements(cls):
        """ get all names of elements in class. Inherited by child classes.
        :return namesOfElements: list of names of elements in this class """
        _elementsInClass = cls.get_all_elements()
        namesOfElements = []
        for _element in _elementsInClass:
            namesOfElements.append(_element.name)
        return namesOfElements
        
    @classmethod
    def get_element(cls,name:str):
        """ get single element in class by name. Inherited by child classes.
        :param name: name of element
        :return element: return element whose name is matched """
        for _element in cls.listOfElements:
            if _element.name == name:
                return _element
        return None

    @classmethod
    def getAllSubclasses(cls):
        """ get all subclasses (child classes) of cls 
        :return subclasses: subclasses of cls """
        return cls.__subclasses__()

    @classmethod
    def get_attribute_of_all_elements(cls,attribute_name:str,capacity_types = False,return_attribute_is_series = False):
        """ get attribute values of all elements in this class 
        :param attribute_name: str name of attribute
        :param capacity_types: boolean if attributes extracted for all capacity types
        :param return_attribute_is_series: boolean if information on attribute type is returned
        :return dictOfAttributes: returns dict of attribute values
        :return attribute_is_series: return information on attribute type """
        system            = EnergySystem.get_system()
        _classElements    = cls.get_all_elements()
        dictOfAttributes  = {}
        attribute_is_series = False
        for _element in _classElements:
            if not capacity_types:
                dictOfAttributes,attribute_is_series = cls.appendAttributeOfElementToDict(_element,attribute_name,dictOfAttributes)
            # if extracted for both capacity types
            else:
                for capacityType in system["setCapacityTypes"]:
                    # append energy only for storage technologies
                    if capacityType == system["setCapacityTypes"][0] or _element.name in system["setStorageTechnologies"]:
                        dictOfAttributes,attribute_is_series = cls.appendAttributeOfElementToDict(_element, attribute_name, dictOfAttributes,capacityType)
        if return_attribute_is_series:
            return dictOfAttributes,attribute_is_series
        else:
            return dictOfAttributes

    @classmethod
    def appendAttributeOfElementToDict(cls,_element,attribute_name,dictOfAttributes,capacityType = None):
        """ get attribute values of all elements in this class
        :param _element: element of class
        :param attribute_name: str name of attribute
        :param dictOfAttributes: dict of attribute values
        :param capacityType: capacity type for which attribute extracted. If None, not listed in key
        :return dictOfAttributes: returns dict of attribute values """
        attribute_is_series = False
        system = EnergySystem.get_system()
        # add Energy for energy capacity type
        if capacityType == system["setCapacityTypes"][1]:
            attribute_name += "Energy"
        assert hasattr(_element, attribute_name), f"Element {_element.name} does not have attribute {attribute_name}"
        _attribute = getattr(_element, attribute_name)
        assert not isinstance(_attribute, pd.DataFrame), f"Not yet implemented for pd.DataFrames. Wrong format for element {_element.name}"
        # add attribute to dictOfAttributes
        if isinstance(_attribute, dict):
            dictOfAttributes.update({(_element.name,)+(key,):val for key,val in _attribute.items()})
        elif isinstance(_attribute, pd.Series) and "PWA" not in attribute_name:
            if capacityType:
                _combinedKey = (_element.name,capacityType)
            else:
                _combinedKey = _element.name
            if len(_attribute) > 1:
                dictOfAttributes[_combinedKey] = _attribute
                attribute_is_series = True
            else:
                dictOfAttributes[_combinedKey] = _attribute.squeeze()
                attribute_is_series = False
            # # if attribute is dict
            # for _key in _attribute:
            #     if isinstance(_key, tuple):
            #         dictOfAttributes[_combinedKey + _key] = _attribute[_key]
            #     else:
            #         dictOfAttributes[_combinedKey + (_key,)] = _attribute[_key]
        elif isinstance(_attribute, int):
            if capacityType:
                dictOfAttributes[(_element.name,capacityType)] = [_attribute]
            else:
                dictOfAttributes[_element.name] = [_attribute]
        else:
            if capacityType:
                dictOfAttributes[(_element.name,capacityType)] = _attribute
            else:
                dictOfAttributes[_element.name] = _attribute
        return dictOfAttributes, attribute_is_series

    @classmethod
    def getAttributeOfSpecificElement(cls,element_name:str,attribute_name:str):
        """ get attribute of specific element in class
        :param element_name: str name of element
        :param attribute_name: str name of attribute
        :return attributeValue: value of attribute"""
        # get element
        _element = cls.get_element(element_name)
        # assert that _element exists and has attribute
        assert _element, f"Element {element_name} not in class {cls}"
        assert hasattr(_element,attribute_name),f"Element {element_name} does not have attribute {attribute_name}"
        attributeValue = getattr(_element,attribute_name)
        return attributeValue

    @classmethod
    def getClassLabel(cls):
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
        EnergySystem.constraint_objective()

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
        for subclass in cls.getAllSubclasses():
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
            name="timeStepsOperationDuration",
            data= EnergySystem.initialize_component(cls,"timeStepsOperationDuration",index_names=["set_elements","setTimeStepsOperation"]),#.astype(int),
            # doc="Parameter which specifies the time step duration in operation for all technologies. Dimensions: set_elements, setTimeStepsOperation"
            doc="Parameter which specifies the time step duration in operation for all technologies"
        )
        # construct pe.Params of the child classes
        for subclass in cls.getAllSubclasses():
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
        for subclass in cls.getAllSubclasses():
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
        for subclass in cls.getAllSubclasses():
            subclass.construct_constraints()

    @classmethod
    def create_custom_set(cls,listIndex):
        """ creates custom set for model component 
        :param listIndex: list of names of indices
        :return custom_set: custom set index
        :return listIndex: list of names of indices """
        listIndexOverwrite = copy.copy(listIndex)
        model           = EnergySystem.get_pyomo_model()
        indexing_sets    = EnergySystem.get_indexing_sets()
        # check if all index sets are already defined in model and no set is indexed
        if all([(hasattr(model,index) and not model.find_component(index).is_indexed()) for index in listIndex]):
            # check if no set is indexed
            listSets = []
            # iterate through indices
            for index in listIndex:
                # if the set already exists in model
                if hasattr(model,index):
                    # append set to list
                    listSets.append(model.find_component(index))
            # return indices as cartesian product of sets
            if len(listIndex) > 1:
                custom_set = list(itertools.product(*listSets))
            else:
                custom_set = list(listSets[0])
            return custom_set,listIndex
        # at least one set is not yet defined
        else:
            # ugly, but if first set is indexingSet
            if listIndex[0] in indexing_sets:
                # empty custom index set
                custom_set = []
                # iterate through
                for element in model.find_component(listIndex[0]):
                    # default: append element
                    appendElement = True
                    # create empty list of sets
                    listSets = []
                    # iterate through indices without first index
                    for index in listIndex[1:]:
                        # if the set already exist in model
                        if hasattr(model,index):
                            # if not indexed
                            if not model.find_component(index).is_indexed():
                                listSets.append(model.find_component(index))
                            # if indexed by first entry
                            elif model.find_component(index).index_set().name in indexing_sets:
                                listSets.append(model.find_component(index)[element])
                            else:
                                raise NotImplementedError
                        # if index is setLocation
                        elif index == "setLocation":
                            # if element in setConversionTechnologies or setStorageTechnologies, append setNodes
                            if element in model.setConversionTechnologies or element in model.setStorageTechnologies:
                                listSets.append(model.setNodes)
                            # if element in setTransportTechnologies
                            elif element in model.setTransportTechnologies:
                                listSets.append(model.setEdges)
                        # if set is built for PWA capex:
                        elif "setCapex" in index:
                            if element in model.setConversionTechnologies:
                                _capexIsPWA = cls.getAttributeOfSpecificElement(element,"capexIsPWA")
                                # if technology is modeled as PWA, break for Linear index
                                if "Linear" in index and _capexIsPWA:
                                    appendElement = False
                                    break
                                # if technology is not modeled as PWA, break for PWA index
                                elif "PWA" in index and not _capexIsPWA:
                                    appendElement = False
                                    break
                            # Transport or Storage technology
                            else:
                                appendElement = False
                                break
                        # if set is built for PWA converEfficiency:
                        elif "setConverEfficiency" in index:
                            if element in model.setConversionTechnologies: # or element in model.setStorageTechnologies:
                                _converEfficiencyIsPWA = cls.getAttributeOfSpecificElement(element, "converEfficiencyIsPWA")
                                dependentCarrier = list(model.setDependentCarriers[element])
                                # TODO for more than one carrier
                                # _PWAConverEfficiency = cls.getAttributeOfSpecificElement(element,"PWAConverEfficiency")
                                # dependentCarrierPWA     = _PWAConverEfficiency["PWAVariables"]
                                if "Linear" in index and not _converEfficiencyIsPWA:
                                    listSets.append(dependentCarrier)
                                elif "PWA" in index and _converEfficiencyIsPWA:
                                    listSets.append(dependentCarrier)
                                else:
                                    listSets.append([])
                                listIndexOverwrite = list(map(lambda x: x.replace(index, 'setCarriers'), listIndex))
                            # Transport or Storage technology
                            else:
                                appendElement = False
                                break
                        # if set is used to determine if on-off behavior is modeled
                        # exclude technologies which have no minLoad and dependentCarrierFlow at referenceCarrierFlow = 0 is also equal to 0
                        elif "OnOff" in index:
                            modelOnOff = cls.checkOnOffModeled(element)
                            if "setNoOnOff" in index:
                                # if modeled as on off, do not append to setNoOnOff
                                if modelOnOff:
                                    appendElement = False
                                    break
                            else:
                                # if not modeled as on off, do not append to setOnOff
                                if not modelOnOff:
                                    appendElement = False
                                    break
                        # split in capacity types of power and energy
                        elif index == "setCapacityTypes":
                            system = EnergySystem.get_system()
                            if element in model.setStorageTechnologies:
                                listSets.append(system["setCapacityTypes"])
                            else:
                                listSets.append([system["setCapacityTypes"][0]])
                        else:
                            raise NotImplementedError(f"Index <{index}> not known")
                    # append indices to custom_set if element is supposed to be appended
                    if appendElement:
                        if listSets:
                            custom_set.extend(list(itertools.product([element],*listSets)))
                        else:
                            custom_set.extend([element])
                return custom_set,listIndexOverwrite
            else:
                raise NotImplementedError

    @classmethod
    def checkOnOffModeled(cls,tech):
        """ this classmethod checks if the on-off-behavior of a technology needs to be modeled.
        If the technology has a minimum load of 0 for all nodes and time steps,
        and all dependent carriers have a lower bound of 0 (only for conversion technologies modeled as PWA),
        then on-off-behavior is not necessary to model
        :param tech: technology in model
        :returns modelOnOff: boolean to indicate that on-off-behavior modeled """
        model = EnergySystem.get_pyomo_model()

        modelOnOff = True
        # check if any min
        _uniqueMinLoad = list(set(cls.getAttributeOfSpecificElement(tech,"minLoad").values))
        # if only one unique minLoad which is zero
        if len(_uniqueMinLoad) == 1 and _uniqueMinLoad[0] == 0:
            # if not a conversion technology, break for current technology
            if tech not in model.setConversionTechnologies:
                modelOnOff = False
            # if a conversion technology, check if all dependentCarrierFlow at referenceCarrierFlow = 0 equal to 0
            else:
                # if technology is approximated (by either PWA or Linear)
                _isPWA = cls.getAttributeOfSpecificElement(tech,"converEfficiencyIsPWA")
                # if not modeled as PWA
                if not _isPWA:
                    modelOnOff = False
                else:
                    _PWAParameter = cls.getAttributeOfSpecificElement(tech,"PWAConverEfficiency")
                    # iterate through all dependent carriers and check if all lower bounds are equal to 0
                    _onlyZeroDependentBound = True
                    for PWAVariable in _PWAParameter["PWAVariables"]:
                        if _PWAParameter["bounds"][PWAVariable][0] != 0:
                            _onlyZeroDependentBound = False
                    if _onlyZeroDependentBound:
                        modelOnOff = False
        # return
        return modelOnOff