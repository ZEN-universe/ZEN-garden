"""===========================================================================================================================================================================
Title:          ENERGY-CARBON OPTIMIZATION PLATFORM
Created:        January-2022
Authors:        Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining a standard EnergySystem. Contains methods to add parameters, variables and constraints to the
                optimization problem. Parent class of the Carrier and Technology classes .The class takes the abstract
                optimization model as an input.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
import pandas as pd
import os
from preprocess.functions.extract_input_data import DataInput

class EnergySystem:
    # energySystem
    energySystem = None
    # pe.ConcreteModel
    concreteModel = None
    # analysis
    analysis = None
    # system
    system = None
    # paths
    paths = None
    # solver
    solver = None

    def __init__(self,nameEnergySystem):
        """ initialization of the energySystem
        :param nameEnergySystem: name of energySystem that is added to the model"""
        # only one energy system can be defined
        assert not EnergySystem.getEnergySystem(), "Only one energy system can be defined."
        # set attributes
        self.name = nameEnergySystem
        # create DataInput object
        self.dataInput = DataInput(EnergySystem.getSystem(),EnergySystem.getAnalysis(),EnergySystem.getSolver(), EnergySystem.getEnergySystem())
        # store input data
        self.storeInputData()
        # add energySystem to list
        EnergySystem.setEnergySystem(self)

    def storeInputData(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """      
        system = EnergySystem.getSystem()
        # in class <EnergySystem>, all sets are constructd
        self.setNodes                   = system["setNodes"]
        self.setNodesOnEdges            = self.dataInput.calculateEdgesFromNodes(self.setNodes)
        self.setEdges                   = list(self.setNodesOnEdges.keys())
        self.setCarriers                = system["setCarriers"]
        self.setTechnologies            = system["setConversionTechnologies"]+system["setTransportTechnologies"]+system["setStorageTechnologies"]
        self.setBaseTimeSteps           = system["setTimeSteps"]
        self.setScenarios               = system["setScenarios"]
        # technology-specific
        self.setConversionTechnologies  = system["setConversionTechnologies"]
        self.setTransportTechnologies   = system["setTransportTechnologies"]
        self.setStorageTechnologies     = system["setStorageTechnologies"]
        
    ### --- classmethods --- ###
    # setter/getter classmethods
    @classmethod
    def setEnergySystem(cls,energySystem):
        """ set energySystem. 
        :param energySystem: new energySystem that is set """
        cls.energySystem = energySystem

    @classmethod
    def setOptimizationAttributes(cls,analysis, system,paths,solver,model):
        """ set attributes of class <EnergySystem> with inputs 
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        :param paths: paths to input folders of data
        :param solver: dictionary defining the solver
        :param model: empty pe.ConcreteModel """
        # set analysis
        cls.analysis = analysis
        # set system
        cls.system = system
        # set input paths
        cls.paths = paths
        # set solver
        cls.solver = solver
        # set concreteModel
        cls.concreteModel = model
        
    @classmethod
    def getConcreteModel(cls):
        """ get concreteModel of the class <EnergySystem>. Every child class can access model and add components.
        :return concreteModel: pe.ConcreteModel """
        return cls.concreteModel

    @classmethod
    def getAnalysis(cls):
        """ get analysis of the class <EnergySystem>. 
        :return analysis: dictionary defining the analysis framework """
        return cls.analysis

    @classmethod
    def getSystem(cls):
        """ get system 
        :return system: dictionary defining the system """
        return cls.system

    @classmethod
    def getPaths(cls):
        """ get paths 
        :return paths: paths to folders of input data """
        return cls.paths
    @classmethod
    def getSolver(cls):
        """ get solver 
        :return solver: dictionary defining the analysis solver """
        return cls.solver

    @classmethod
    def getEnergySystem(cls):
        """ get energySystem.
        :return energySystem: return energySystem  """
        return cls.energySystem
    
    @classmethod
    def getAttribute(cls,attributeName:str):
        """ get attribute value of energySystem
        :param attributeName: str name of attribute
        :return attribute: returns attribute values """
        energySystem = cls.getEnergySystem()
        assert hasattr(energySystem,attributeName), f"The energy system does not have attribute '{attributeName}"
        return getattr(energySystem,attributeName)
    
    @classmethod
    def calculateTimeStepDuration(cls,inputTimeSteps):
        """ calculates (equidistant) time step durations for input time steps
        :param inputTimeSteps: input time steps
        :return timeStepDurationDict: dict with duration of each time step """
        baseTimeSteps = cls.getEnergySystem().setBaseTimeSteps
        durationInputTimeSteps = len(baseTimeSteps)/len(inputTimeSteps)
        timeStepDurationDict = {timeStep: durationInputTimeSteps for timeStep in inputTimeSteps}
        return timeStepDurationDict

    @classmethod
    def decodeTimeStep(cls,element:str,elementTimeStep:int,timeStepType:str = None):
        """ decodes timeStep, i.e., retrieves the baseTimeStep corresponding to the variablTimeStep of a element.
        timeStep of element --> baseTimeStep of model 
        :param element: element of model, i.e., carrier or technology
        :param elementTimeStep: time step of element
        :param timeStepType: invest or operation. Only relevant for technologies, None for carrier
        :return baseTimeStep: baseTimeStep of model """
        model = cls.getConcreteModel()
        # get time step duration
        if not timeStepType:
            timeStepDuration = model.timeStepsCarrierDuration
        elif timeStepType == "invest":
            timeStepDuration = model.timeStepsInvestDuration
        elif timeStepType == "operation":
            timeStepDuration = model.timeStepsOperationDuration
        else:
            raise KeyError(f"time step type {timeStepType} is invalid. Only 'invest', 'operation', or None accepted.")
        # calculate timeSteps from the beginning
        baseTimeStep = sum([timeStepDuration[element,timeStep] for timeStep in range(1,elementTimeStep+1)])
        # mainly for debugging, check if baseTimeStep is integer
        assert baseTimeStep.is_integer(),f"The element time step {elementTimeStep} of element {element} does not correspond to an integer baseTimeStep ({baseTimeStep})"
        return(int(baseTimeStep))

    @classmethod
    def encodeTimeStep(cls,element:str,baseTimeStep:int,timeStepType:str = None):
        """ encodes baseTimeStep, i.e., retrieves the time step of a element corresponding to baseTimeStep of model.
        baseTimeStep of model --> timeStep of element 
        :param element: element of model, i.e., carrier or technology
        :param baseTimeStep: base time step of model for which the corresponding time index is extracted
        :param timeStepType: invest or operation. Only relevant for technologies
        :return outputTimeStep: time step of element"""
        model = cls.getConcreteModel()
        # get time step duration
        if not timeStepType:
            timeStepDuration = model.timeStepsCarrierDuration
        elif timeStepType == "invest":
            timeStepDuration = model.timeStepsInvestDuration
        elif timeStepType == "operation":
            timeStepDuration = model.timeStepsOperationDuration
        else:
            raise KeyError(f"time step type {timeStepType} is invalid. Only 'invest', 'operation', or None accepted.")
        # calculate summed elementTimeStep
        elementTimeStep = 0
        summedDuration = 0
        # sum the time step durations until summedDuration >= baseTimeStep
        while summedDuration < baseTimeStep:
            elementTimeStep += 1
            summedDuration += timeStepDuration[element,elementTimeStep]
        return(elementTimeStep)

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to EnergySystem --- ###
    @classmethod
    def constructSets(cls):
        """ constructs the pe.Sets of the class <EnergySystem> """
        # construct pe.Sets of the class <EnergySystem>
        model = cls.getConcreteModel()
        energySystem = cls.getEnergySystem()

        # nodes
        model.setNodes = pe.Set(
            initialize=energySystem.setNodes, 
            doc='Set of nodes')
        # connected nodes
        model.setAliasNodes = pe.Set(
            initialize=energySystem.setNodes,
            doc='Copy of the set of nodes to model edges. Subset: setNodes')
        # edges
        model.setEdges = pe.Set(
            initialize = energySystem.setEdges,
            doc = 'Set of edges'
        )
        # nodes on edges
        model.setNodesOnEdges = pe.Set(
            model.setEdges,
            initialize = energySystem.setNodesOnEdges,
            doc = 'Set of nodes that constitute an edge. Edge connects first node with second node.'
        )
        # carriers
        model.setCarriers = pe.Set(
            initialize=energySystem.setCarriers,
            doc='Set of carriers')
        # technologies
        model.setTechnologies = pe.Set(
            initialize=energySystem.setTechnologies,
            doc='Set of technologies')
        # time-steps
        model.setBaseTimeSteps = pe.Set(
            initialize=energySystem.setBaseTimeSteps,
            doc='Set of base time-steps')
        # scenarios
        model.setScenarios = pe.Set(
            initialize=energySystem.setScenarios,
            doc='Set of scenarios')

    @classmethod
    def constructParams(cls):
        """ constructs the pe.Params of the class <EnergySystem> """
        # currently no pe.Params in the class <EnergySystem>
        pass

    @classmethod
    def constructVars(cls):
        """ constructs the pe.Vars of the class <EnergySystem> """
        # currently no pe.Vars in the class <EnergySystem>
        pass

    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <EnergySystem> """
        # currently no pe.Constraints in the class <EnergySystem>
        pass
    
    @classmethod
    def constructObjective(cls):
        """ constructs the pe.Objective of the class <EnergySystem> """
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

        # construct objective
        model.objective = pe.Objective(
            rule = objectiveRule,
            sense = objectiveSense
        )

# different objective
def objectiveTotalCostRule(model):
    """objective function to minimize the total cost"""
    # CARRIERS
    carrierImport = sum(
                        sum(
                            sum(model.importCarrierFlow[carrier, node, time] * model.importPriceCarrier[carrier, node, time] * model.timeStepsCarrierDuration[carrier, time]
                            for time in model.setTimeStepsCarrier[carrier])
                        for node in model.setNodes)
                    for carrier in model.setCarriers)

    carrierExport = sum(
                        sum(
                            sum(model.exportCarrierFlow[carrier, node, time] * model.exportPriceCarrier[carrier, node, time] * model.timeStepsCarrierDuration[carrier, time]
                            for time in model.setTimeStepsCarrier[carrier])
                        for node in model.setNodes)
                    for carrier in model.setCarriers)

    return(carrierImport - carrierExport + model.capexTotal)

def objectiveCarbonEmissionsRule(model):
    """objective function to minimize total emissions"""

    # TODO implement objective functions for emissions
    return pe.Constraint.Skip

def objectiveRiskRule(model):
    """objective function to minimize total risk"""

    # TODO implement objective functions for risk
    return pe.Constraint.Skip

