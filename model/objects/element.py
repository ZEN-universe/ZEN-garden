"""===========================================================================================================================================================================
Title:          ENERGY-CARBON OPTIMIZATION PLATFORM
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining a standard element. Contains methods to add parameters, variables and constraints to the
                optimization problem. Parent class of the Carrier and Technology classes .The class takes the abstract
                optimization model as an input.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
import pandas as pd
import os
from preprocess.functions.calculate_input_data import DataInput

class Element:
    # empty list of elements
    listOfElements = []
    # pe.ConcreteModel
    concreteModel = None
    # analysis
    analysis = None
    # system
    system = None
    # paths
    paths = None

    def __init__(self,element):
        """ initialization of an element
        :param element: element that is added to the model"""
        # set attributes
        self.name = element
        # add element to list
        Element.addElement(self)
        # create DataInput object
        self.dataInput = DataInput(Element.getSystem(),Element.getAnalysis(),Element.getElement("grid"))

    def storeInputData(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """      
        system = Element.getSystem()
        # in class <Element>, all sets are defined
        self.setNodes                   = system["setNodes"]
        self.setNodesOnEdges            = DataInput.calculateEdgesFromNodes(self.setNodes)
        self.setEdges                   = list(self.setNodesOnEdges.keys())
        self.setCarriers                = system["setCarriers"]
        self.setTechnologies            = system["setTechnologies"]
        self.setTimeSteps               = system["setTimeSteps"]
        self.setScenarios               = system["setScenarios"]
        # carrier-specific
        self.setImportCarriers          = system["setImportCarriers"]
        self.setExportCarriers          = system["setExportCarriers"]
        # technology-specific
        self.setConversionTechnologies  = system["setConversionTechnologies"]
        self.setTransportTechnologies   = system["setTransportTechnologies"]
        
    ### --- classmethods --- ###
    # setter/getter classmethods
    @classmethod
    def addElement(cls,element):
        """ add element to element list. Inherited by child classes.
        :param element: new element that is to be added to the list """
        cls.listOfElements.append(element)

    @classmethod
    def setOptimizationAttributes(cls,analysis, system,paths,model):
        """ set attributes of class <Element> with inputs 
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        :param paths: paths to input folders of data
        :param model: empty pe.ConcreteModel """
        # set analysis
        cls.analysis = analysis
        # set system
        cls.system = system
        # set input paths
        cls.paths = paths
        # set concreteModel
        cls.concreteModel = model
        
    @classmethod
    def getConcreteModel(cls):
        """ get concreteModel of the class <Element>. Every child class can access model and add components.
        :return concreteModel: pe.ConcreteModel """
        return cls.concreteModel

    @classmethod
    def getAnalysis(cls):
        """ get analysis of the class <Element>. 
        :return analysis: pe.analysis """
        return cls.analysis

    @classmethod
    def getSystem(cls):
        """ get system 
        :return system: input dictionary of optimization """
        return cls.system

    @classmethod
    def getPaths(cls):
        """ get paths 
        :return paths: paths to folders of input data """
        return cls.paths

    @classmethod
    def getAllElements(cls):
        """ get all elements in class. Inherited by child classes.
        :return cls.listOfElements: list of elements in this class """
        return cls.listOfElements

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
    def getAttributeOfElement(cls,elementName:str,attributeName:str):
        """ get attribute value of single element in this class 
        :param elementName: str name of element
        :param attributeName: str name of attribute
        :return attribute: returns attribute values """
        _element = cls.getElement(elementName)
        assert _element,"Class {} does not have instance for element '{}'".format(cls,elementName)
        if hasattr(_element,attributeName):
            _attribute = getattr(_element,attributeName)
            return _attribute
        else:
            raise AttributeError("Element '{}' does not have attribute '{}'".format(elementName,attributeName))

    @classmethod
    def getAttributeOfAllElements(cls,attributeName:str):
        """ get attribute values of all elements in this class 
        :param attributeName: str name of attribute
        :return dictOfAttributes: returns dict of attribute values """
        _classElements = cls.getAllElements()
        dictOfAttributes = {}
        for _element in _classElements:
            if hasattr(_element,attributeName):
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

    

    ### --- classmethods to define sets, parameters, variables, and constraints, that correspond to Element --- ###
    # Here, after defining Element-specific components, the components of the other classes are defined
    @classmethod
    def defineModelComponents(cls):
        """ defines the model components of the class <Element> """
        # define pe.Sets
        cls.defineSets()
        # define pe.Params
        cls.defineParams()
        # define pe.Vars
        cls.defineVars()
        # define pe.Constraints
        cls.defineConstraints()
        # define pe.Objective
        cls.defineObjective()

    @classmethod
    def defineSets(cls):
        """ defines the pe.Sets of the class <Element> """
        # define pe.Sets of the class <Element>
        model = cls.getConcreteModel()
        grid = cls.getElement("grid")

        # nodes
        model.setNodes = pe.Set(
            initialize=grid.setNodes, 
            doc='Set of nodes')
        # connected nodes
        model.setAliasNodes = pe.Set(
            initialize=grid.setNodes,
            doc='Copy of the set of nodes to model edges. Subset: setNodes')
        # edges
        model.setEdges = pe.Set(
            initialize = grid.setEdges,
            doc = 'Set of edges'
        )
        # nodes on edges
        model.setNodesOnEdges = pe.Set(
            model.setEdges,
            initialize = grid.setNodesOnEdges,
            doc = 'Set of nodes that constitute an edge. Edge connects first node with second node.'
        )
        # carriers
        model.setCarriers = pe.Set(
            initialize=grid.setCarriers,
            doc='Set of carriers')
        # technologies
        model.setTechnologies = pe.Set(
            initialize=grid.setTechnologies,
            doc='Set of technologies')
        # time-steps
        model.setTimeSteps = pe.Set(
            initialize=grid.setTimeSteps,
            doc='Set of time-steps')
        # scenarios
        model.setScenarios = pe.Set(
            initialize=grid.setScenarios,
            doc='Set of scenarios')

        # define pe.Sets of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.defineSets()

    @classmethod
    def defineParams(cls):
        """ defines the pe.Params of the class <Element> """
        # define pe.Params of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.defineParams()

    @classmethod
    def defineVars(cls):
        """ defines the pe.Vars of the class <Element> """
        # define pe.Vars of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.defineVars()

    @classmethod
    def defineConstraints(cls):
        """ defines the pe.Constraints of the class <Element> """
        # define pe.Constraints of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.defineConstraints()
    
    @classmethod
    def defineObjective(cls):
        """ defines the pe.Objective of the class <Element> """
        # get model
        model = cls.getConcreteModel()

        # get selected objective rule
        if cls.getAnalysis()["objective"] == "TotalCost":
            objectiveRule = objectiveTotalCostRule
        elif cls.getAnalysis()["objective"] == "CarbonEmissions":
            logging.info("Objective of carbon emissions not yet implemented")
            objectiveRule = objectiveCarbonEmissionsRule
        elif cls.getAnalysis()["objective"] == "Risk":
            logging.info("Objective of carbon emissions not yet implemented")
            objectiveRule = objectiveRiskRule
        else:
            logging.error("Objective type {} not known".format(cls.getAnalysis()["objective"]))

        # get selected objective sense
        if cls.getAnalysis()["sense"] == "minimize":
            objectiveSense = pe.minimize
        elif cls.getAnalysis()["sense"] == "maximize":
            objectiveSense = pe.maximize
        else:
            logging.error("Objective sense {} not known".format(cls.getAnalysis()["sense"]))

        # define objective
        model.objective = pe.Objective(
            rule = objectiveRule,
            sense = objectiveSense
        )

# different objective
def objectiveTotalCostRule(model):
    """objective function to minimize the total cost"""

    # CARRIERS
    carrierImport = sum(sum(sum(model.importCarrierFlow[carrier, node, time] * model.importPriceCarrier[carrier, node, time]
                            for time in model.setTimeSteps)
                        for node in model.setNodes)
                    for carrier in model.setImportCarriers)

    carrierExport = sum(sum(sum(model.exportCarrierFlow[carrier, node, time] * model.exportPriceCarrier[carrier, node, time]
                            for time in model.setTimeSteps)
                        for node in model.setNodes)
                    for carrier in model.setExportCarriers)

    return(carrierImport - carrierExport + model.capexTotalTechnology)

def objectiveCarbonEmissionsRule(model):
    """objective function to minimize total emissions"""

    # TODO implement objective functions for emissions
    return pe.Constraint.Skip

def objectiveRiskRule(model):
    """objective function to minimize total risk"""

    # TODO implement objective functions for risk
    return pe.Constraint.Skip

