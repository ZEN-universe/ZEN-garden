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
import numpy as np
import pandas as pd
from pint import UnitRegistry
from preprocess.functions.extract_input_data import DataInput
from pint.util import column_echelon_form
import warnings

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
    # unit registry
    ureg = UnitRegistry()
    # empty list of indexing sets
    indexingSets = []
    # aggregationObjects of element
    aggregationObjectsOfElements = {}
    # empty dict of technologies of carrier
    dictTechnologyOfCarrier = {}
    # empty dict of order of time steps operation
    dictOrderTimeStepsOperation = {}
    # empty dict of order of time steps invest
    dictOrderTimeStepsInvest = {}
    # empty dict of order of time steps yearly
    dictOrderTimeStepsYearly = {}
    # empty dict of raw time series, only necessary for single time grid approach
    dictTimeSeriesRaw = {}

    def __init__(self,nameEnergySystem):
        """ initialization of the energySystem
        :param nameEnergySystem: name of energySystem that is added to the model"""
        # only one energy system can be defined
        assert not EnergySystem.getEnergySystem(), "Only one energy system can be defined."
        # set attributes
        self.name       = nameEnergySystem
        # add energySystem to list
        EnergySystem.setEnergySystem(self)
        # create DataInput object
        self.dataInput  = DataInput(EnergySystem.getSystem(),EnergySystem.getAnalysis(),EnergySystem.getSolver(), EnergySystem.getEnergySystem())
        # store input data
        self.storeInputData()

    def storeInputData(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """      
        system                          = EnergySystem.getSystem()
        self.paths                      = EnergySystem.getPaths()
        self.getBaseUnits()
        # in class <EnergySystem>, all sets are constructed
        self.setNodes                   = self.dataInput.extractLocations()
        self.setNodesOnEdges            = self.calculateEdgesFromNodes()
        self.setEdges                   = list(self.setNodesOnEdges.keys())
        self.setCarriers                = system["setCarriers"]
        self.setTechnologies            = system["setConversionTechnologies"] + system["setTransportTechnologies"] + system["setStorageTechnologies"]
        self.setScenarios               = system["setScenarios"]
        # base time steps
        self.setBaseTimeSteps           = list(range(0,system["timeStepsPerYear"]*system["timeStepsYearly"]))
        self.setBaseTimeStepsYearly     = list(range(0, system["timeStepsPerYear"]))
        # yearly time steps
        self.typesTimeSteps             = ["invest", "operation", "yearly"]
        self.dictNumberOfTimeSteps      = self.dataInput.extractNumberTimeSteps()
        self.setTimeStepsYearly         = self.dataInput.extractTimeSteps(typeOfTimeSteps="yearly")
        self.timeStepsYearlyDuration    = EnergySystem.calculateTimeStepDuration(self.setTimeStepsYearly)
        self.orderTimeStepsYearly       = np.concatenate([[timeStep] * self.timeStepsYearlyDuration[timeStep] for timeStep in self.timeStepsYearlyDuration])
        self.setOrderTimeSteps(None, self.orderTimeStepsYearly, timeStepType="yearly")
        # technology-specific
        self.setConversionTechnologies  = system["setConversionTechnologies"]
        self.setTransportTechnologies   = system["setTransportTechnologies"]
        self.setStorageTechnologies     = system["setStorageTechnologies"]
        # carbon emissions limit
        self.carbonEmissionsLimit       = self.dataInput.extractInputData(self.paths["setScenarios"]["folder"], "carbonEmissionsLimit", indexSets=["setTimeSteps"], timeSteps=self.setTimeStepsYearly)
        _fractionOfYear                 = system["timeStepsPerYear"]/system["totalHoursPerYear"]
        self.carbonEmissionsLimit       = self.carbonEmissionsLimit*_fractionOfYear # reduce to fraction of year

    def getBaseUnits(self):
        """ gets base units of energy system """
        _listBaseUnits                  = self.dataInput.extractBaseUnits(self.paths["setScenarios"]["folder"])
        ureg                            = EnergySystem.getUnitRegistry()
        # load additional units
        ureg.load_definitions(self.paths["setScenarios"]["folder"]+"/unitDefinitions.txt")
        # empty base units and dimensionality matrix
        self.baseUnits                  = {}
        self.dimMatrix                  = pd.DataFrame(index=_listBaseUnits).astype(int)
        for _baseUnit in _listBaseUnits:
            dimUnit                     = ureg.get_dimensionality(ureg(_baseUnit))
            self.baseUnits[_baseUnit]   = ureg(_baseUnit).dimensionality
            self.dimMatrix.loc[_baseUnit, list(dimUnit.keys())] = list(dimUnit.values())
        self.dimMatrix                  = self.dimMatrix.fillna(0).astype(int).T
        # check if unit defined twice or more
        _duplicateUnits                 = self.dimMatrix.T.duplicated()
        if _duplicateUnits.any():
            _dimMatrixDuplicate         = self.dimMatrix.loc[:,_duplicateUnits]
            for _duplicate in _dimMatrixDuplicate:
                # if same unit twice (same order of magnitude and same dimensionality)
                if len(self.dimMatrix[_duplicate].shape) > 1:
                    logging.warning(f"The base unit <{_duplicate}> was defined more than once. Duplicates are dropped.")
                    _duplicateDim               = self.dimMatrix[_duplicate].T.drop_duplicates().T
                    self.dimMatrix              = self.dimMatrix.drop(_duplicate,axis=1)
                    self.dimMatrix[_duplicate]  = _duplicateDim
                else:
                    raise KeyError(f"More than one base unit defined for dimensionality {self.baseUnits[_duplicate]} (e.g., {_duplicate})")
        # get linearly dependent units
        M, I, pivot                     = column_echelon_form(np.array(self.dimMatrix), ntype=float)
        M                               = np.array(M).squeeze()
        I                               = np.array(I).squeeze()
        pivot                           = np.array(pivot).squeeze()
        # index of linearly dependent units in M and I
        idxLinDep                       = np.squeeze(np.argwhere(np.all(M==0,axis=1)))
        # index of linearly dependent units in dimensionality matrix
        _idxPivot                           = range(len(self.baseUnits))
        idxLinDepDimMatrix                  = list(set(_idxPivot).difference(pivot))
        self.dimAnalysis                    = {}
        self.dimAnalysis["dependentUnits"]  = self.dimMatrix.columns[idxLinDepDimMatrix]
        dependentDims                       = I[idxLinDep,:]
        # if only one dependent unit
        if len(self.dimAnalysis["dependentUnits"]) == 1:
            dependentDims                   = dependentDims.reshape(1,dependentDims.size)
        # reorder dependent dims to match dependent units
        DimOfDependentUnits                 = dependentDims[:,idxLinDepDimMatrix]
        # if not already in correct order (ones on the diagonal of dependentDims)
        if not np.all(np.diag(DimOfDependentUnits)==1):
            # get position of ones in DimOfDependentUnits
            posOnes         = np.argwhere(DimOfDependentUnits==1)
            assert np.size(posOnes,axis=0) == len(self.dimAnalysis["dependentUnits"]), \
                f"Cannot determine order of dependent base units {self.dimAnalysis['dependentUnits']}, " \
                f"because diagonal of dimensions of the dependent units cannot be determined."
            # pivot dependent dims
            dependentDims   = dependentDims[posOnes[:,1],:]
        self.dimAnalysis["dependentDims"]   = dependentDims
        # check that no base unit can be directly constructed from the others (e.g., GJ from GW and hour)
        assert ~DataInput.checkIfPosNegBoolean(dependentDims,axis=1), f"At least one of the base units {list(self.baseUnits.keys())} can be directly constructed from the others"

    def calculateEdgesFromNodes(self):
        """ calculates setNodesOnEdges from setNodes
        :return setNodesOnEdges: dict with edges and corresponding nodes """
        setNodesOnEdges = {}
        # read edge file
        setEdgesInput = self.dataInput.extractLocations(extractNodes=False)
        if setEdgesInput is not None:
            for edge in setEdgesInput.index:
                setNodesOnEdges[edge] = (setEdgesInput.loc[edge,"nodeFrom"],setEdgesInput.loc[edge,"nodeTo"])
        else:
            warnings.warn("Implicit creation of edges will be deprecated. Provide 'setEdges.csv' in folder 'setNodes' instead!",FutureWarning)
            for nodeFrom in self.setNodes:
                for nodeTo in self.setNodes:
                    if nodeFrom != nodeTo:
                        setNodesOnEdges[nodeFrom+"-"+nodeTo] = (nodeFrom,nodeTo)
        return setNodesOnEdges

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
        # set indexing sets
        cls.setIndexingSets()
    
    @classmethod
    def setIndexingSets(cls):
        """ set sets that serve as an index for other sets """
        system = cls.getSystem()
        # iterate over sets
        for key in system:
            if "set" in key:
                cls.indexingSets.append(key)

    @classmethod
    def setTechnologyOfCarrier(cls,technology,listTechnologyOfCarrier):
        """ appends technology to carrier in dictTechnologyOfCarrier
        :param technology: name of technology in model
        :param listTechnologyOfCarrier: list of carriers correspondent to technology"""
        for carrier in listTechnologyOfCarrier:
            if carrier not in cls.dictTechnologyOfCarrier:
                cls.dictTechnologyOfCarrier[carrier] = [technology]
            elif technology not in cls.dictTechnologyOfCarrier[carrier]:
                cls.dictTechnologyOfCarrier[carrier].append(technology)

    @classmethod
    def setOrderTimeSteps(cls,element,orderTimeSteps,timeStepType = None):
        """ sets order of time steps, either of operation, invest, or year
        :param element: name of element in model
        :param orderTimeSteps: list of time steps corresponding to base time step
        :param timeStepType: type of time step (operation, invest or year)"""
        if not timeStepType:
            timeStepType = "operation"

        if timeStepType == "operation":
            cls.dictOrderTimeStepsOperation[element] = orderTimeSteps
        elif timeStepType == "invest":
            cls.dictOrderTimeStepsInvest[element] = orderTimeSteps
        elif timeStepType == "yearly":
            cls.dictOrderTimeStepsYearly[element] = orderTimeSteps
        else:
            raise KeyError(f"Time step type {timeStepType} is incorrect")

    @classmethod
    def setAggregationObjects(cls,element,aggregationObject):
        """ append aggregation object of element
        :param element: element in model 
        :param aggregationObject: object of TimeSeriesAggregation"""
        cls.aggregationObjectsOfElements[element] = aggregationObject

    @classmethod
    def setTimeSeriesRaw(cls,aggregationObject):
        """ appends the raw time series of elements
        :param aggregationObject: object of TimeSeriesAggregation """
        cls.dictTimeSeriesRaw[aggregationObject.element] = aggregationObject.dfTimeSeriesRaw

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
    def getUnitRegistry(cls):
        """ get the unit registry
        :return units: unit registry """
        return cls.ureg

    @classmethod
    def getAttribute(cls,attributeName:str):
        """ get attribute value of energySystem
        :param attributeName: str name of attribute
        :return attribute: returns attribute values """
        energySystem = cls.getEnergySystem()
        assert hasattr(energySystem,attributeName), f"The energy system does not have attribute '{attributeName}"
        return getattr(energySystem,attributeName)

    @classmethod
    def getIndexingSets(cls):
        """ set sets that serve as an index for other sets 
        :return cls.indexingSets: list of sets that serve as an index for other sets"""
        return cls.indexingSets

    @classmethod
    def getTechnologyOfCarrier(cls,carrier):
        """ gets technologies which are connected by carrier 
        :param carrier: carrier which connects technologies
        :return listOfTechnologies: list of technologies connected by carrier"""
        if carrier in cls.dictTechnologyOfCarrier:
            return cls.dictTechnologyOfCarrier[carrier]
        else:
            return None

    @classmethod
    def getOrderTimeSteps(cls,element,timeStepType = None):
        """ get order ot time steps of element
        :param element: name of element in model
        :param timeStepType: type of time step (operation or invest)
        :return orderTimeSteps: list of time steps corresponding to base time step"""
        if not timeStepType:
            timeStepType = "operation"
        if timeStepType == "operation":
            return cls.dictOrderTimeStepsOperation[element]
        elif timeStepType == "invest":
            return cls.dictOrderTimeStepsInvest[element]
        elif timeStepType == "yearly":
            return cls.dictOrderTimeStepsYearly[element]
        else:
            raise KeyError(f"Time step type {timeStepType} is incorrect")

    @classmethod
    def getAggregationObjects(cls,element):
        """ get aggregation object of element
        :param element: element in model 
        :return aggregationObject: object of TimeSeriesAggregation """
        return cls.aggregationObjectsOfElements[element] 

    @classmethod
    def getTimeSeriesRaw(cls,element):
        """ get the raw time series of element
        :param element: element in model 
        :return dfTimeSeriesRaw: raw time series of element """
        if element in cls.dictTimeSeriesRaw:
            return cls.dictTimeSeriesRaw[element]
        else:
            return None

    @classmethod
    def calculateConnectedEdges(cls,node,direction:str):
        """ calculates connected edges going in (direction = 'in') or going out (direction = 'out') 
        :param node: current node, connected by edges 
        :param direction: direction of edges, either in or out. In: node = endnode, out: node = startnode
        :return setConnectedEdges: list of connected edges """
        energySystem = cls.getEnergySystem()
        if direction == "in":
            # second entry is node into which the flow goes
            setConnectedEdges = [edge for edge in energySystem.setNodesOnEdges if energySystem.setNodesOnEdges[edge][1]==node]
        elif direction == "out":
            # first entry is node out of which the flow starts
            setConnectedEdges = [edge for edge in energySystem.setNodesOnEdges if energySystem.setNodesOnEdges[edge][0]==node]
        else:
            raise KeyError(f"invalid direction '{direction}'")
        return setConnectedEdges

    @classmethod
    def calculateTimeStepDuration(cls,inputTimeSteps,manualBaseTimeSteps = None):
        """ calculates (equidistant) time step durations for input time steps
        :param inputTimeSteps: input time steps
        :param manualBaseTimeSteps: manual list of base time steps
        :return timeStepDurationDict: dict with duration of each time step """
        if manualBaseTimeSteps is not None:
            baseTimeSteps       = manualBaseTimeSteps
        else:
            baseTimeSteps       = cls.getEnergySystem().setBaseTimeSteps
        durationInputTimeSteps  = len(baseTimeSteps)/len(inputTimeSteps)
        timeStepDurationDict    = {timeStep: int(durationInputTimeSteps) for timeStep in inputTimeSteps}
        if not durationInputTimeSteps.is_integer():
            logging.warning(f"The duration of each time step {durationInputTimeSteps} of input time steps {inputTimeSteps} does not evaluate to an integer. \n"
                            f"The duration of the last time step is set to compensate for the difference")
            durationLastTimeStep = len(baseTimeSteps) - sum(timeStepDurationDict[key] for key in timeStepDurationDict if key != inputTimeSteps[-1])
            timeStepDurationDict[inputTimeSteps[-1]] = durationLastTimeStep
        return timeStepDurationDict

    @classmethod
    def decodeTimeStep(cls,element:str,elementTimeStep:int,timeStepType:str = None):
        """ decodes timeStep, i.e., retrieves the baseTimeStep corresponding to the variableTimeStep of a element.
        timeStep of element --> baseTimeStep of model 
        :param element: element of model, i.e., carrier or technology
        :param elementTimeStep: time step of element
        :param timeStepType: invest or operation. Only relevant for technologies, None for carrier
        :return baseTimeStep: baseTimeStep of model """
        orderTimeSteps = cls.getOrderTimeSteps(element,timeStepType)
        # find where elementTimeStep in order of element time steps
        baseTimeSteps = np.argwhere(orderTimeSteps == elementTimeStep)
        return baseTimeSteps

    @classmethod
    def encodeTimeStep(cls,element:str,baseTimeSteps:int,timeStepType:str = None,yearly=False):
        """ encodes baseTimeStep, i.e., retrieves the time step of a element corresponding to baseTimeStep of model.
        baseTimeStep of model --> timeStep of element 
        :param element: element of model, i.e., carrier or technology
        :param baseTimeStep: base time step of model for which the corresponding time index is extracted
        :param timeStepType: invest or operation. Only relevant for technologies
        :return outputTimeStep: time step of element"""
        # model = cls.getConcreteModel()
        orderTimeSteps = cls.getOrderTimeSteps(element,timeStepType)
        # get time step duration
        elementTimeStep = np.unique(orderTimeSteps[baseTimeSteps])
        if yearly:
            return(elementTimeStep)
        if len(elementTimeStep) == 1:
            return(elementTimeStep[0])
        else:
            raise LookupError(f"Currently only implemented for a single element time step, not {elementTimeStep}")

    @classmethod
    def convertTechnologyTimeStepType(cls,element,elementTimeStep,direction = "operation2invest"):
        """ converts type of technology time step from operation to invest or from invest to operation.
        Carrier has no invest, so irrelevant for carrier
        :param element: element of model (here technology)
        :param elementTimeStep: time step of element
        :param direction: conversion direction (operation2invest or invest2operation)
        :return convertedTimeStep: time of second type """
        model                   = cls.getConcreteModel()
        setTimeStepsInvest      = model.setTimeStepsInvest[element]
        setTimeStepsOperation   = model.setTimeStepsOperation[element]
        # if only one investment step
        if len(setTimeStepsInvest) == 1:
            if direction ==  "operation2invest":
                return setTimeStepsInvest.at(1)
            elif direction == "invest2operation":
                return setTimeStepsOperation.data()
            else:
                raise KeyError(f"Direction for time step conversion {direction} is incorrect")
        # if more than one invest step
        else:
            if direction ==  "operation2invest":
                orderTimeStepsIn        = cls.getOrderTimeSteps(element,"operation")
                orderTimeStepsOut       = cls.getOrderTimeSteps(element,"invest")
            elif direction == "invest2operation":
                orderTimeStepsOut       = cls.getOrderTimeSteps(element,"operation")
                orderTimeStepsIn        = cls.getOrderTimeSteps(element,"invest")
            else:
                raise KeyError(f"Direction for time step conversion {direction} is incorrect")
            # convert time steps
            convertedTimeSteps = np.unique(orderTimeStepsOut[orderTimeStepsIn == elementTimeStep])
            assert len(convertedTimeSteps) == 1, f"more than one converted time step, not yet implemented"
            return convertedTimeSteps[0]

    @classmethod
    def getLifetimeRange(cls, tech, time,timeStepType:str = None):
        """ returns lifetime range of technology. If timeStepType, then converts the yearly time step 'time' to timeStepType """
        model               = cls.getConcreteModel()
        if timeStepType:
            baseTimeSteps   = cls.decodeTimeStep(None,time,"yearly")
            investTimeStep  = cls.encodeTimeStep(tech,baseTimeSteps,timeStepType,yearly=True)
            a=1
        else:
            investTimeStep  = time
        tStart,tEnd         = cls.getStartEndTimeOfLifetime(tech,investTimeStep)

        return range(tStart,tEnd+1)

    @classmethod
    def getStartEndTimeOfLifetime(cls,tech,investTimeStep):
        """ counts back the lifetime to get the start invest time step and returns startInvestTimeStep """
        # get model and system
        model               = cls.getConcreteModel()
        system              = cls.getSystem()
        # get endInvestTimeStep
        if not isinstance(investTimeStep,np.ndarray):
            endInvestTimeStep = investTimeStep
        elif len(investTimeStep) == 1:
            endInvestTimeStep = investTimeStep[0]
        # if more than one investment time step
        else:
            endInvestTimeStep = investTimeStep[-1]
            investTimeStep = investTimeStep[0]
        # decode to base time steps
        baseTimeSteps       = cls.decodeTimeStep(tech,investTimeStep,timeStepType="invest")
        baseTimeStep        = baseTimeSteps[0]
        # convert lifetime to interval of base time steps
        baseLifetime        = model.lifetimeTechnology[tech]/system["intervalYears"]*system["timeStepsPerYear"]
        if int(baseLifetime) != baseLifetime:
            logging.warning(f"The lifetime of {tech} does not translate to an integer lifetime interval in the base time domain ({baseLifetime})")
        startBaseTimeStep   = int(max(0,baseTimeStep-baseLifetime + 1))
        startInvestTimeStep = cls.encodeTimeStep(tech,startBaseTimeStep,timeStepType="invest",yearly=True)[0]
        return startInvestTimeStep,endInvestTimeStep

    @classmethod
    def getFullTimeSeriesOfComponent(cls,component,indexSubsets:tuple,manualOrderName = None):
        """ returns full time series of result component 
        :param component: component (parameter or variable) of optimization model 
        :param indexSubsets: dict of index subsets {<levelOfSubset>:<value(s)OfSubset>} 
        :return fullTimeSeries: full time series """
        # TODO quick fix
        if manualOrderName:
            orderName   = manualOrderName
        else:
            orderName   = indexSubsets[0]
        _componentData  = component.extract_values()
        dfData          = pd.Series(_componentData,index=_componentData.keys())
        dfReducedData   = dfData.loc[indexSubsets]
        orderElement    = EnergySystem.getOrderTimeSteps(orderName)
        fullTimeSeries  = np.zeros(np.size(orderElement))
        for timeStep in dfReducedData.index:
            fullTimeSeries[orderElement==timeStep] = dfReducedData[timeStep]
        return fullTimeSeries

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to EnergySystem --- ###
    @classmethod
    def constructSets(cls):
        """ constructs the pe.Sets of the class <EnergySystem> """
        # construct pe.Sets of the class <EnergySystem>
        model           = cls.getConcreteModel()
        energySystem    = cls.getEnergySystem()

        # nodes
        model.setNodes = pe.Set(
            initialize=energySystem.setNodes,
            doc='Set of nodes')
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
        # yearly time steps
        model.setTimeStepsYearly = pe.Set(
            initialize=energySystem.setTimeStepsYearly,
            doc='Set of yearly time-steps')

    @classmethod
    def constructParams(cls):
        """ constructs the pe.Params of the class <EnergySystem> """
        # get model
        model = cls.getConcreteModel()

        # carbon emissions limit
        model.carbonEmissionsLimit = pe.Param(
            model.setTimeStepsYearly,
            initialize = cls.getEnergySystem().carbonEmissionsLimit,
            doc = 'Parameter which specifies the total limit on carbon emissions'
        )

    @classmethod
    def constructVars(cls):
        """ constructs the pe.Vars of the class <EnergySystem> """
        # get model
        model = cls.getConcreteModel()

        # carbon emissions
        model.carbonEmissionsTotal = pe.Var(
            model.setTimeStepsYearly,
            domain = pe.NonNegativeReals,
            doc = "total carbon emissions of energy system. Domain: NonNegativeReals"
        )

    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <EnergySystem> """
        # get model
        model = cls.getConcreteModel()

        # carbon emissions
        model.constraintCarbonEmissionsTotal = pe.Constraint(
            model.setTimeStepsYearly,
            rule = constraintCarbonEmissionsTotalRule,
            doc = "total carbon emissions of energy system"
        )
        # carbon emissions
        model.constraintCarbonEmissionsLimit = pe.Constraint(
            model.setTimeStepsYearly,
            rule=constraintCarbonEmissionsLimitRule,
            doc="limit of total carbon emissions of energy system"
        )
    
    @classmethod
    def constructObjective(cls):
        """ constructs the pe.Objective of the class <EnergySystem> """
        logging.info("Construct pe.Objective")
        # get model
        model = cls.getConcreteModel()

        # get selected objective rule
        if cls.getAnalysis()["objective"] == "TotalCost":
            objectiveRule = objectiveTotalCostRule
        elif cls.getAnalysis()["objective"] == "TotalCarbonEmissions":
            objectiveRule = objectiveTotalCarbonEmissionsRule
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
            rule    = objectiveRule,
            sense   = objectiveSense
        )

def constraintCarbonEmissionsTotalRule(model,year):
    """ add up all carbon emissions from technologies and carriers """
    return(
        model.carbonEmissionsTotal[year] ==
        # technologies
        model.carbonEmissionsTechnologyTotal[year]
        + 
        # carriers
        model.carbonEmissionsCarrierTotal[year]
    )

def constraintCarbonEmissionsLimitRule(model, year):
    """ time dependent carbon emissions limit from technologies and carriers"""
    if model.carbonEmissionsLimit[year] != np.inf:
        return(
            model.carbonEmissionsLimit[year] >= model.carbonEmissionsTotal[year]
        )
    else:
        return pe.Constraint.Skip

# objective rules
def objectiveTotalCostRule(model):
    """objective function to minimize the total cost"""
    return(
            sum(
                model.capexTotal[year] + model.opexTotal[year] + model.costCarrierTotal[year]
            for year in model.setTimeStepsYearly)
    )

def objectiveTotalCarbonEmissionsRule(model):
    """objective function to minimize total emissions"""
    return(sum(model.carbonEmissionsTotal[year] for year in model.setTimeStepsYearly))

def objectiveRiskRule(model):
    """objective function to minimize total risk"""
    # TODO implement objective functions for risk
    return pe.Constraint.Skip

