"""===========================================================================================================================================================================
Title:          ENERGY-CARBON OPTIMIZATION PLATFORM
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining a standard Element. Contains methods to add parameters, variables and constraints to the
                optimization problem. Parent class of the Carrier and Technology classes .The class takes the abstract
                optimization model as an input.
==========================================================================================================================================================================="""
import itertools 
import logging
import numpy as np
from preprocess.functions.extract_input_data import DataInput
from model.objects.energy_system import EnergySystem

class Element:
    # empty list of elements
    listOfElements = []

    def __init__(self,element):
        """ initialization of an element
        :param element: element that is added to the model"""
        # set attributes
        self.name = element
        # create DataInput object
        self.dataInput = DataInput(EnergySystem.getSystem(),EnergySystem.getAnalysis(),EnergySystem.getSolver(), EnergySystem.getEnergySystem())
        # add element to list
        Element.addElement(self)
        
    ### --- classmethods --- ###
    # setter/getter classmethods
    @classmethod
    def addElement(cls,element):
        """ add element to element list. Inherited by child classes.
        :param element: new element that is to be added to the list """
        cls.listOfElements.append(element)

    @classmethod
    def getAllElements(cls):
        """ get all elements in class. Inherited by child classes.
        :return cls.listOfElements: list of elements in this class """
        return cls.listOfElements

    @classmethod
    def getAllNamesOfElements(cls):
        """ get all names of elements in class. Inherited by child classes.
        :return namesOfElements: list of names of elements in this class """
        _elementsInClass = cls.getAllElements()
        namesOfElements = []
        for _element in _elementsInClass:
            namesOfElements.append(_element.name)
        return namesOfElements
        
    @classmethod
    def getElement(cls,name:str):
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
    def getAttributeOfAllElements(cls,attributeName:str):
        """ get attribute values of all elements in this class 
        :param attributeName: str name of attribute
        :return dictOfAttributes: returns dict of attribute values """
        _classElements = cls.getAllElements()
        dictOfAttributes = {}
        for _element in _classElements:
            assert hasattr(_element,attributeName),f"Element {_element} does not have attribute {attributeName}"
            _attribute = getattr(_element,attributeName)
            if isinstance(_attribute,dict):
                # if attribute is dict
                for _key in _attribute:
                    if isinstance(_key,tuple):
                        dictOfAttributes[(_element.name,)+_key] = _attribute[_key]
                    else:
                        dictOfAttributes[(_element.name,_key)] = _attribute[_key]
            else:
                dictOfAttributes[_element.name] = _attribute

        return dictOfAttributes

    @classmethod
    def getAttributeOfSpecificElement(cls,elementName:str,attributeName:str):
        """ get attribute of specific element in class
        :param elementName: str name of element
        :param attributeName: str name of attribute
        :return attributeValue: value of attribute"""
        # get element
        _element = cls.getElement(elementName)
        # assert that _element exists and has attribute
        assert _element, f"Element {elementName} not in class {cls}"
        assert hasattr(_element,attributeName),f"Element {_element} does not have attribute {attributeName}"
        attributeValue = getattr(_element,attributeName)
        return attributeValue

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Element --- ###
    # Here, after defining EnergySystem-specific components, the components of the other classes are constructed
    @classmethod
    def constructModelComponents(cls):
        """ constructs the model components of the class <Element> """
        logging.info("\n--- Construct model components ---\n")
        # construct pe.Sets
        cls.constructSets()
        # construct pe.Params
        cls.constructParams()
        # construct pe.Vars
        cls.constructVars()
        # construct pe.Constraints
        cls.constructConstraints()
        # construct pe.Objective
        EnergySystem.constructObjective()

    @classmethod
    def constructSets(cls):
        """ constructs the pe.Sets of the class <Element> """
        logging.info("Construct pe.Sets")
        # construct pe.Sets of energy system
        EnergySystem.constructSets()
        # construct pe.Sets of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.constructSets()

    @classmethod
    def constructParams(cls):
        """ constructs the pe.Params of the class <Element> """
        logging.info("Construct pe.Params")
        # construct pe.Params of energy system
        EnergySystem.constructParams()
        # construct pe.Params of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.constructParams()

    @classmethod
    def constructVars(cls):
        """ constructs the pe.Vars of the class <Element> """
        logging.info("Construct pe.Vars")
        # construct pe.Vars of energy system
        EnergySystem.constructVars()
        # construct pe.Vars of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.constructVars()

    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <Element> """
        logging.info("Construct pe.Constraints")
        # construct pe.Constraints of energy system
        EnergySystem.constructConstraints()
        # construct pe.Constraints of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.constructConstraints()

    @classmethod
    def createCustomSet(cls,listIndex):
        """ creates custom set for model component 
        :param listIndex: list of names of indices
        :return customSet: custom set index """
        model = EnergySystem.getConcreteModel()
        # indexingSets = ["setTechnologies", "setConversionTechnologies", "setTransportTechnologies", "setCarriers","setPWACapexTechs","setNLCapexTechs","setPWAConverEfficiencyTechs","setNLConverEfficiencyTechs"]
        indexingSets = EnergySystem.getIndexingSets()
        # check if all index sets are already defined in model and no set is indexed
        if all([(hasattr(model,index) and not model.find_component(index).is_indexed()) for index in listIndex]):
            # check if no set is indexed
            listSets = []
            # iterate through indices
            for index in listIndex:
                # if the set already exist in model
                if hasattr(model,index):
                    # append set to list
                    listSets.append(model.find_component(index))
            # return indices as cartesian product of sets
            customSet = list(itertools.product(*listSets))
            return customSet
        # at least one set is not yet defined
        else:
            # ugly, but if first set is indexingSet
            if listIndex[0] in indexingSets:
                # empty custom index set
                customSet = []
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
                            elif model.find_component(index).index_set().name in indexingSets:
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
                            if element in model.setConversionTechnologies: # TODO or element in model.setStorageTechnologies:
                                # if technology is approximated (by either PWA or Linear)
                                if element not in EnergySystem.getAnalysis()["nonlinearTechnologyApproximation"]["Capex"] or EnergySystem.getSolver()["model"] == "MILP":
                                    _PWAParameter = cls.getAttributeOfSpecificElement(element,"PWAParameter")
                                    # if technology is modeled as PWA, break for Linear index
                                    if "Linear" in index and "capex" in _PWAParameter["Capex"]["PWAVariables"]:
                                        appendElement = False
                                        break
                                    # if technology is not modeled as PWA, break for PWA index
                                    elif "PWA" in index and "capex" not in _PWAParameter["Capex"]["PWAVariables"]:
                                        appendElement = False
                                        break
                                    # if NL
                                    elif "NL" in index:
                                        appendElement = False
                                        break
                                # if technology is not approximated (i.e., modeled as NL), break for approximated index 
                                else:
                                    if "NL" not in index:
                                        appendElement = False
                                        break
                            # Transport technology
                            else:
                                appendElement = False
                                break
                        # if set is built for PWA converEfficiency:
                        elif "setConverEfficiency" in index:
                            if element in model.setConversionTechnologies: # or element in model.setStorageTechnologies:
                                # if technology is approximated (by either PWA or Linear)
                                if element not in EnergySystem.getAnalysis()["nonlinearTechnologyApproximation"]["ConverEfficiency"] or EnergySystem.getSolver()["model"] == "MILP":
                                    _PWAParameter = cls.getAttributeOfSpecificElement(element,"PWAParameter")
                                    dependentCarrier = model.setDependentCarriers[element]
                                    dependentCarrierPWA = _PWAParameter["ConverEfficiency"]["PWAVariables"]
                                    if "Linear" in index:
                                        listSets.append(dependentCarrier-dependentCarrierPWA)
                                    elif "PWA" in index:
                                        listSets.append(dependentCarrierPWA)
                                    # if NL
                                    elif "NL" in index:
                                        appendElement = False
                                        break  
                                    # if approximated, either PWA or Linear
                                    else:
                                        listSets.append(dependentCarrier)
                                # if technology is not approximated (hence modeled as NL), break for approximated index 
                                else:
                                    if "NL" not in index:
                                        appendElement = False
                                        break
                            # Transport technology
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
                        else:
                            raise NotImplementedError
                    # append indices to customSet if element is supposed to be appended
                    if appendElement:
                        if listSets:
                            customSet.extend(list(itertools.product([element],*listSets)))
                        else:
                            customSet.extend([element])
                return customSet
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
        model = EnergySystem.getConcreteModel()

        modelOnOff = True
        # check if any min
        _uniqueMinLoad = list(set(cls.getAttributeOfSpecificElement(tech,"minLoad").values()))
        # if only one unique minLoad which is zero
        if len(_uniqueMinLoad) == 1 and _uniqueMinLoad[0] == 0:
            # if not a conversion technology, break for current technology
            if tech not in model.setConversionTechnologies:
                modelOnOff = False
            # if a conversion technology, check if all dependentCarrierFlow at referenceCarrierFlow = 0 equal to 0
            else:
                # if technology is approximated (by either PWA or Linear)
                if tech not in EnergySystem.getAnalysis()["nonlinearTechnologyApproximation"]["ConverEfficiency"] or EnergySystem.getSolver()["model"] == "MILP":
                    _PWAParameter = cls.getAttributeOfSpecificElement(tech,"PWAParameter")["ConverEfficiency"]
                    # if not modeled as PWA
                    if not _PWAParameter["PWAVariables"]:
                        modelOnOff = False
                    # iterate through all dependent carriers and check if all lower bounds are equal to 0
                    _onlyZeroDependentBound = True
                    for PWAVariable in _PWAParameter["PWAVariables"]:
                        if _PWAParameter["bounds"][PWAVariable][0] != 0:
                            _onlyZeroDependentBound = False
                    if _onlyZeroDependentBound:
                        modelOnOff = False
        # return
        return modelOnOff