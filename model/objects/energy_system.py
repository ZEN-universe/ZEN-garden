"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        January-2022
Authors:        Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining a standard EnergySystem. Contains methods to add parameters, variables and constraints to the
                optimization problem. Parent class of the Carrier and Technology classes .The class takes the abstract
                optimization model as an input.
==========================================================================================================================================================================="""
import logging
import warnings
import pyomo.environ as pe
import numpy         as np
import pandas        as pd
import copy
from pint                                    import UnitRegistry
from preprocess.functions.extract_input_data import DataInput
from preprocess.functions.unit_handling         import UnitHandling

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
    # unit handling instance
    unitHandling = None
    # empty list of indexing sets
    indexingSets = []
    # empty dict of technologies of carrier
    dictTechnologyOfCarrier = {}
    # empty dict of sequence of time steps operation
    dictSequenceTimeStepsOperation = {}
    # empty dict of sequence of time steps invest
    dictSequenceTimeStepsInvest = {}
    # empty dict of sequence of time steps yearly
    dictSequenceTimeStepsYearly = {}
    # empty dict of conversion from energy time steps to power time steps for storage technologies
    dictTimeStepsEnergy2Power = {}
    # empty dict of conversion from operational time steps to invest time steps for technologies
    dictTimeStepsOperation2Invest = {}
    # empty dict of matching the last time step of the year in the storage domain to the first
    dictTimeStepsStorageLevelStartEndYear = {}
    # empty dict of element classes
    dictElementClasses = {}
    # empty list of class names
    elementList = {}

    def __init__(self,nameEnergySystem):
        """ initialization of the energySystem
        :param nameEnergySystem: name of energySystem that is added to the model """

        # only one energy system can be defined
        assert not EnergySystem.getEnergySystem(), "Only one energy system can be defined."

        # set attributes
        self.name = nameEnergySystem

        # add energySystem to list
        EnergySystem.setEnergySystem(self)

        # get input path
        self.getInputPath()

        # create UnitHandling object
        EnergySystem.createUnitHandling()

        # create DataInput object
        self.dataInput = DataInput(self,EnergySystem.getSystem(), EnergySystem.getAnalysis(), EnergySystem.getSolver(), EnergySystem.getEnergySystem(), self.unitHandling)

        # store input data
        self.storeInputData()

    def storeInputData(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """

        system                          = EnergySystem.getSystem()
        self.paths                      = EnergySystem.getPaths()

        # in class <EnergySystem>, all sets are constructed
        self.setNodes               = self.dataInput.extractLocations()
        self.setNodesOnEdges        = self.calculateEdgesFromNodes()
        self.setEdges               = list(self.setNodesOnEdges.keys())
        self.setCarriers            = []
        self.setTechnologies        = system["setTechnologies"]
        # base time steps
        self.setBaseTimeSteps       = list(range(0, system["unaggregatedTimeStepsPerYear"] * system["optimizedYears"]))
        self.setBaseTimeStepsYearly = list(range(0, system["unaggregatedTimeStepsPerYear"]))

        # yearly time steps
        self.typesTimeSteps                  = ["invest", "operation", "yearly"]
        self.dictNumberOfTimeSteps           = self.dataInput.extractNumberTimeSteps()
        self.setTimeStepsYearly              = self.dataInput.extractTimeSteps(typeOfTimeSteps="yearly")
        self.setTimeStepsYearlyEntireHorizon = copy.deepcopy(self.setTimeStepsYearly)
        self.timeStepsYearlyDuration         = EnergySystem.calculateTimeStepDuration(self.setTimeStepsYearly)
        self.sequenceTimeStepsYearly         = np.concatenate([[timeStep] * self.timeStepsYearlyDuration[timeStep] for timeStep in self.timeStepsYearlyDuration])
        self.setSequenceTimeSteps(None, self.sequenceTimeStepsYearly, timeStepType="yearly")

        # technology-specific
        self.setConversionTechnologies = system["setConversionTechnologies"]
        self.setTransportTechnologies  = system["setTransportTechnologies"]
        self.setStorageTechnologies    = system["setStorageTechnologies"]
        # carbon emissions limit
        self.carbonEmissionsLimit    = self.dataInput.extractInputData("carbonEmissionsLimit", indexSets=["setTimeSteps"],
                                                                    timeSteps=self.setTimeStepsYearly)
        _fractionOfYear              = system["unaggregatedTimeStepsPerYear"] / system["totalHoursPerYear"]
        self.carbonEmissionsLimit    = self.carbonEmissionsLimit * _fractionOfYear  # reduce to fraction of year
        self.carbonEmissionsBudget   = self.dataInput.extractInputData("carbonEmissionsBudget", indexSets=[])
        self.previousCarbonEmissions = self.dataInput.extractInputData("previousCarbonEmissions", indexSets=[])
        # carbon price
        self.carbonPrice = self.dataInput.extractInputData("carbonPrice", indexSets=["setTimeSteps"],
                                                           timeSteps=self.setTimeStepsYearly)
        self.carbonPriceOvershoot = self.dataInput.extractInputData("carbonPriceOvershoot", indexSets=[])

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

    def getInputPath(self):
        """ get input path where input data is stored inputPath"""
        folderLabel = EnergySystem.getSystem()["folderNameSystemSpecification"]

        paths = EnergySystem.getPaths()
        # get input path of energy system specification
        self.inputPath = paths[folderLabel]["folder"]

    ### CLASS METHODS ###
    # setter/getter classmethods
    @classmethod
    def setEnergySystem(cls,energySystem):
        """ set energySystem. 
        :param energySystem: new energySystem that is set """
        cls.energySystem = energySystem

    @classmethod
    def setOptimizationAttributes(cls,analysis, system,paths,solver):
        """ set attributes of class <EnergySystem> with inputs 
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        :param paths: paths to input folders of data
        :param solver: dictionary defining the solver"""
        # set analysis
        cls.analysis = analysis
        # set system
        cls.system = system
        # set input paths
        cls.paths = paths
        # set solver
        cls.solver = solver
        # set indexing sets
        cls.setIndexingSets()

    @classmethod
    def setConcreteModel(cls,concreteModel):
        """ sets empty concrete model to energySystem
        :param concreteModel: pe.ConcreteModel"""
        cls.concreteModel = concreteModel

    @classmethod
    def setIndexingSets(cls):
        """ set sets that serve as an index for other sets """
        system = cls.getSystem()
        # iterate over sets
        for key in system:
            if "set" in key:
                cls.indexingSets.append(key)

    @classmethod
    def setManualSetToIndexingSets(cls,set):
        """ manually set to cls.indexingSets """
        cls.indexingSets.append(set)

    @classmethod
    def setTechnologyOfCarrier(cls,technology,listTechnologyOfCarrier):
        """ appends technology to carrier in dictTechnologyOfCarrier
        :param technology: name of technology in model
        :param listTechnologyOfCarrier: list of carriers correspondent to technology"""
        for carrier in listTechnologyOfCarrier:
            if carrier not in cls.dictTechnologyOfCarrier:
                cls.dictTechnologyOfCarrier[carrier] = [technology]
                cls.energySystem.setCarriers.append(carrier)
            elif technology not in cls.dictTechnologyOfCarrier[carrier]:
                cls.dictTechnologyOfCarrier[carrier].append(technology)

    @classmethod
    def setTimeStepsEnergy2Power(cls, element, timeStepsEnergy2Power):
        """ sets the dict of converting the energy time steps to the power time steps of storage technologies """
        cls.dictTimeStepsEnergy2Power[element] = timeStepsEnergy2Power

    @classmethod
    def setTimeStepsOperation2Invest(cls, element, timeStepsOperation2Invest):
        """ sets the dict of converting the operational time steps to the invest time steps of all technologies """
        cls.dictTimeStepsOperation2Invest[element] = timeStepsOperation2Invest

    @classmethod
    def setTimeStepsStorageStartEnd(cls, element):
        """ sets the dict of matching the last time step of the year in the storage level domain to the first """
        system = cls.getSystem()
        _baseTimeStepsPerYear = system["unaggregatedTimeStepsPerYear"]
        _numberYears = system["optimizedYears"]
        _sequenceTimeSteps = cls.getSequenceTimeSteps(element + "StorageLevel")
        _timeStepsStart = _sequenceTimeSteps[np.array(range(0, _numberYears)) * _baseTimeStepsPerYear]
        _timeStepsEnd = _sequenceTimeSteps[np.array(range(1, _numberYears + 1)) * _baseTimeStepsPerYear - 1]
        cls.dictTimeStepsStorageLevelStartEndYear[element] = {_start: _end for _start, _end in
                                                              zip(_timeStepsStart, _timeStepsEnd)}

    @classmethod
    def setSequenceTimeSteps(cls,element,sequenceTimeSteps,timeStepType = None):
        """ sets sequence of time steps, either of operation, invest, or year
        :param element: name of element in model
        :param sequenceTimeSteps: list of time steps corresponding to base time step
        :param timeStepType: type of time step (operation, invest or year)"""
        if not timeStepType:
            timeStepType = "operation"

        if timeStepType == "operation":
            cls.dictSequenceTimeStepsOperation[element] = sequenceTimeSteps
        elif timeStepType == "invest":
            cls.dictSequenceTimeStepsInvest[element]    = sequenceTimeSteps
        elif timeStepType == "yearly":
            cls.dictSequenceTimeStepsYearly[element]    = sequenceTimeSteps
        else:
            raise KeyError(f"Time step type {timeStepType} is incorrect")

    @classmethod
    def setSequenceTimeStepsDict(cls,dictAllSequenceTimeSteps):
        """ sets all dicts of sequences of time steps.
        :param dictAllSequenceTimeSteps: dict of all dictSequenceTimeSteps"""
        cls.dictSequenceTimeStepsOperation = dictAllSequenceTimeSteps["operation"]
        cls.dictSequenceTimeStepsInvest    = dictAllSequenceTimeSteps["invest"]
        cls.dictSequenceTimeStepsYearly    = dictAllSequenceTimeSteps["yearly"]

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
        :return energySystem: return energySystem """
        return cls.energySystem

    @classmethod
    def getElementList(cls):
        """ get attribute value of energySystem
        :param attributeName: str name of attribute
        :return attribute: returns attribute values """
        elementClasses    = cls.dictElementClasses.keys()
        carrierClasses    = [elementName for elementName in elementClasses if "Carrier" in elementName]
        technologyClasses = [elementName for elementName in elementClasses if "Technology" in elementName]
        cls.elementList   = technologyClasses + carrierClasses
        return cls.elementList

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
    def getTimeStepsEnergy2Power(cls, element):
        """ gets the dict of converting the energy time steps to the power time steps of storage technologies """
        return cls.dictTimeStepsEnergy2Power[element]

    @classmethod
    def getTimeStepsOperation2Invest(cls, element):
        """ gets the dict of converting the operational time steps to the invest time steps of technologies """
        return cls.dictTimeStepsOperation2Invest[element]

    @classmethod
    def getTimeStepsStorageStartEnd(cls, element, timeStep):
        """ gets the dict of converting the operational time steps to the invest time steps of technologies """
        if timeStep in cls.dictTimeStepsStorageLevelStartEndYear[element].keys():
            return cls.dictTimeStepsStorageLevelStartEndYear[element][timeStep]
        else:
            return None

    @classmethod
    def getSequenceTimeSteps(cls,element,timeStepType = None):
        """ get sequence ot time steps of element
        :param element: name of element in model
        :param timeStepType: type of time step (operation or invest)
        :return sequenceTimeSteps: list of time steps corresponding to base time step"""
        if not timeStepType:
            timeStepType = "operation"
        if timeStepType == "operation":
            return cls.dictSequenceTimeStepsOperation[element]
        elif timeStepType == "invest":
            return cls.dictSequenceTimeStepsInvest[element]
        elif timeStepType == "yearly":
            return cls.dictSequenceTimeStepsYearly[element]
        else:
            raise KeyError(f"Time step type {timeStepType} is incorrect")

    @classmethod
    def getSequenceTimeStepsDict(cls):
        """ returns all dicts of sequence of time steps.
        :return dictAllSequenceTimeSteps: dict of all dictSequenceTimeSteps"""
        dictAllSequenceTimeSteps = {
            "operation" : cls.dictSequenceTimeStepsOperation,
            "invest"    : cls.dictSequenceTimeStepsInvest,
            "yearly"    : cls.dictSequenceTimeStepsYearly
        }
        return dictAllSequenceTimeSteps

    @classmethod
    def getUnitHandling(cls):
        """ returns the unit handling object """
        return cls.unitHandling

    @classmethod
    def createUnitHandling(cls):
        """ creates and stores the unit handling object """
        # create UnitHandling object
        cls.unitHandling = UnitHandling(cls.getEnergySystem().inputPath,cls.getEnergySystem().solver["roundingDecimalPoints"])

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
    def calculateReversedEdge(cls, edge):
        """ calculates the reversed edge corresponding to an edge
        :param edge: input edge
        :return reversedEdge: edge which corresponds to the reversed direction of edge"""
        energySystem = cls.getEnergySystem()
        nodeOut, nodeIn = energySystem.setNodesOnEdges[edge]
        for reversedEdge in energySystem.setNodesOnEdges:
            if nodeOut == energySystem.setNodesOnEdges[reversedEdge][1] and nodeIn == \
                    energySystem.setNodesOnEdges[reversedEdge][0]:
                return reversedEdge
        raise KeyError(f"Edge {edge} has no reversed edge. However, at least one transport technology is bidirectional")

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
    def decodeTimeStep(cls,element,elementTimeStep:int,timeStepType:str = None):
        """ decodes timeStep, i.e., retrieves the baseTimeStep corresponding to the variableTimeStep of a element.
        timeStep of element --> baseTimeStep of model
        :param element: element of model, i.e., carrier or technology
        :param elementTimeStep: time step of element
        :param timeStepType: invest or operation. Only relevant for technologies, None for carrier
        :return baseTimeStep: baseTimeStep of model """
        sequenceTimeSteps = cls.getSequenceTimeSteps(element,timeStepType)
        # find where elementTimeStep in sequence of element time steps
        baseTimeSteps = np.argwhere(sequenceTimeSteps == elementTimeStep)
        return baseTimeSteps

    @classmethod
    def encodeTimeStep(cls,element:str,baseTimeSteps:int,timeStepType:str = None,yearly=False):
        """ encodes baseTimeStep, i.e., retrieves the time step of a element corresponding to baseTimeStep of model.
        baseTimeStep of model --> timeStep of element
        :param element: name of element in model, i.e., carrier or technology
        :param baseTimeStep: base time step of model for which the corresponding time index is extracted
        :param timeStepType: invest or operation. Only relevant for technologies
        :return outputTimeStep: time step of element"""
        # model = cls.getConcreteModel()
        sequenceTimeSteps = cls.getSequenceTimeSteps(element,timeStepType)
        # get time step duration
        if np.all(baseTimeSteps >= 0):
            elementTimeStep = np.unique(sequenceTimeSteps[baseTimeSteps])
        else:
            elementTimeStep = [-1]
        if yearly:
            return(elementTimeStep)
        if len(elementTimeStep) == 1:
            return(elementTimeStep[0])
        else:
            raise LookupError(f"Currently only implemented for a single element time step, not {elementTimeStep}")

    @classmethod
    def decodeYearlyTimeSteps(cls,elementTimeSteps):
        """ decodes list of years to base time steps
        :param elementTimeSteps: time steps of year
        :return fullBaseTimeSteps: full list of time steps """
        listBaseTimeSteps = []
        for year in elementTimeSteps:
            listBaseTimeSteps.append(cls.decodeTimeStep(None,year,"yearly"))
        fullBaseTimeSteps = np.concatenate(listBaseTimeSteps)
        return fullBaseTimeSteps

    @classmethod
    def convertTimeStepEnergy2Power(cls,element,timeStepEnergy):
        """ converts the time step of the energy quantities of a storage technology to the time step of the power quantities """
        _timeStepsEnergy2Power = cls.getTimeStepsEnergy2Power(element)
        return _timeStepsEnergy2Power[timeStepEnergy]

    @classmethod
    def convertTimeStepOperation2Invest(cls, element, timeStepOperation):
        """ converts the operational time step to the invest time step """
        _timeStepsOperation2Invest = cls.getTimeStepsOperation2Invest(element)
        return _timeStepsOperation2Invest[timeStepOperation]

    @classmethod
    def initializeComponent(cls,callingClass,componentName,indexNames = None,setTimeSteps = None,capacityTypes = False):
        """ this method initializes a modeling component by extracting the stored input data.
        :param callingClass: class from where the method is called
        :param componentName: name of modeling component
        :param indexNames: names of index sets, only if callingClass is not EnergySystem
        :param setTimeSteps: time steps, only if callingClass is EnergySystem
        :return componentData: data to initialize the component """

        # if calling class is EnergySystem
        if callingClass == cls:
            component = getattr(cls.getEnergySystem(), componentName)
            if setTimeSteps:
                componentData = component[setTimeSteps]
            elif type(component) == float:
                componentData = component
            else:
                componentData = component.squeeze()
        else:
            component = callingClass.getAttributeOfAllElements(componentName, capacityTypes)
            componentData = pd.Series(component, dtype=float)
            if indexNames:
                customSet = callingClass.createCustomSet(indexNames)
                componentData = componentData[customSet]
        return componentData

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
        # all elements
        model.setElements = pe.Set(
            initialize=model.setTechnologies | model.setCarriers,
            doc='Set of elements')
        # set setElements to indexingSets
        cls.setManualSetToIndexingSets("setElements")
        # time-steps
        model.setBaseTimeSteps = pe.Set(
            initialize=energySystem.setBaseTimeSteps,
            doc='Set of base time-steps')
        # yearly time steps
        model.setTimeStepsYearly = pe.Set(
            initialize=energySystem.setTimeStepsYearly,
            doc='Set of yearly time-steps')
        # yearly time steps of entire optimization horizon
        model.setTimeStepsYearlyEntireHorizon = pe.Set(
            initialize=energySystem.setTimeStepsYearlyEntireHorizon,
            doc='Set of yearly time-steps of entire optimization horizon')

    @classmethod
    def constructParams(cls):
        """ constructs the pe.Params of the class <EnergySystem> """
        # get model
        model = cls.getConcreteModel()

        # carbon emissions limit
        model.carbonEmissionsLimit = pe.Param(
            model.setTimeStepsYearly,
            initialize = cls.initializeComponent(cls,"carbonEmissionsLimit", setTimeSteps =model.setTimeStepsYearly),
            doc = 'Parameter which specifies the total limit on carbon emissions'
        )
        # carbon emissions budget
        model.carbonEmissionsBudget = pe.Param(
            initialize=cls.initializeComponent(cls, "carbonEmissionsBudget"),
            doc='Parameter which specifies the total budget of carbon emissions until the end of the entire time horizon'
        )
        # carbon emissions budget
        model.previousCarbonEmissions = pe.Param(
            initialize=cls.initializeComponent(cls, "previousCarbonEmissions"),
            doc='Parameter which specifies the total previous carbon emissions'
        )
        # carbon price
        model.carbonPrice = pe.Param(
            model.setTimeStepsYearly,
            initialize=cls.initializeComponent(cls, "carbonPrice", setTimeSteps=model.setTimeStepsYearly),
            doc='Parameter which specifies the yearly carbon price'
        )
        # carbon price of overshoot
        model.carbonPriceOvershoot = pe.Param(
            initialize=cls.initializeComponent(cls, "carbonPriceOvershoot"),
            doc='Parameter which specifies the carbon price for budget overshoot'
        )

    @classmethod
    def constructVars(cls):
        """ constructs the pe.Vars of the class <EnergySystem> """
        # get model
        model = cls.getConcreteModel()

        # carbon emissions
        model.carbonEmissionsTotal = pe.Var(
            model.setTimeStepsYearly,
            domain = pe.Reals,
            doc = "total carbon emissions of energy system. Domain: Reals"
        )
        # carbon emissions
        model.carbonEmissionsCumulative = pe.Var(
            model.setTimeStepsYearly,
            domain=pe.Reals,
            doc="cumulative carbon emissions of energy system over time for each year. Domain: Reals"
        )
        # carbon emissions
        model.carbonEmissionsOvershoot = pe.Var(
            model.setTimeStepsYearly,
            domain=pe.NonNegativeReals,
            doc="overshoot carbon emissions of energy system at the end of the time horizon. Domain: Reals"
        )
        # cost of carbon emissions
        model.costCarbonEmissionsTotal = pe.Var(
            model.setTimeStepsYearly,
            domain=pe.Reals,
            doc="total cost of carbon emissions of energy system. Domain: Reals"
        )
        # costs
        model.costTotal = pe.Var(
            model.setTimeStepsYearly,
            domain=pe.Reals,
            doc="total cost of energy system. Domain: Reals"
        )
        # NPV
        model.NPV = pe.Var(
            model.setTimeStepsYearly,
            domain=pe.Reals,
            doc="NPV of energy system. Domain: Reals"
        )

    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <EnergySystem> """
        # get model
        model = cls.getConcreteModel()

        # carbon emissions
        model.constraintCarbonEmissionsTotal = pe.Constraint(
            model.setTimeStepsYearly,
            rule=constraintCarbonEmissionsTotalRule,
            doc="total carbon emissions of energy system"
        )
        # carbon emissions
        model.constraintCarbonEmissionsCumulative = pe.Constraint(
            model.setTimeStepsYearly,
            rule=constraintCarbonEmissionsCumulativeRule,
            doc="cumulative carbon emissions of energy system over time"
        )
        # cost of carbon emissions
        model.constraintCarbonCostTotal = pe.Constraint(
            model.setTimeStepsYearly,
            rule=constraintCarbonCostTotalRule,
            doc="total carbon cost of energy system"
        )
        # carbon emissions
        model.constraintCarbonEmissionsLimit = pe.Constraint(
            model.setTimeStepsYearly,
            rule=constraintCarbonEmissionsLimitRule,
            doc="limit of total carbon emissions of energy system"
        )
        # carbon emissions
        model.constraintCarbonEmissionsBudget = pe.Constraint(
            model.setTimeStepsYearly,
            rule=constraintCarbonEmissionsBudgetRule,
            doc="Budget of total carbon emissions of energy system"
        )
        # costs
        model.constraintCostTotal = pe.Constraint(
            model.setTimeStepsYearly,
            rule=constraintCostTotalRule,
            doc="total cost of energy system"
        )
        # NPV
        model.constraintNPV = pe.Constraint(
            model.setTimeStepsYearly,
            rule=constraintNPVRule,
            doc="NPV of energy system"
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
            rule=objectiveRule,
            sense=objectiveSense
        )

def constraintCarbonEmissionsTotalRule(model, year):
    """ add up all carbon emissions from technologies and carriers """
    return (
            model.carbonEmissionsTotal[year] ==
            # technologies
            model.carbonEmissionsTechnologyTotal[year]
            +
            # carriers
            model.carbonEmissionsCarrierTotal[year]
    )

def constraintCarbonEmissionsCumulativeRule(model, year):
    """ cumulative carbon emissions over time """
    intervalBetweenYears = EnergySystem.getSystem()["intervalBetweenYears"]
    if year == model.setTimeStepsYearly.at(1):
        return (
                model.carbonEmissionsCumulative[year] ==
                model.carbonEmissionsTotal[year]
                + model.previousCarbonEmissions
        )
    else:
        return (
                model.carbonEmissionsCumulative[year] ==
                model.carbonEmissionsCumulative[year - 1]
                + model.carbonEmissionsTotal[year - 1] * (intervalBetweenYears - 1)
                + model.carbonEmissionsTotal[year]
        )

def constraintCarbonCostTotalRule(model, year):
    """ carbon cost associated with the carbon emissions of the system in each year """
    return (
            model.costCarbonEmissionsTotal[year] ==
            model.carbonPrice[year] * model.carbonEmissionsTotal[year]
            # add overshoot price
            + model.carbonEmissionsOvershoot[year] * model.carbonPriceOvershoot
    )

def constraintCarbonEmissionsLimitRule(model, year):
    """ time dependent carbon emissions limit from technologies and carriers"""
    if model.carbonEmissionsLimit[year] != np.inf:
        return (
                model.carbonEmissionsLimit[year] >= model.carbonEmissionsTotal[year]
        )
    else:
        return pe.Constraint.Skip

def constraintCarbonEmissionsBudgetRule(model, year):
    """ carbon emissions budget of entire time horizon from technologies and carriers.
    The prediction extends until the end of the horizon, i.e.,
    last optimization time step plus the current carbon emissions until the end of the horizon """
    intervalBetweenYears = EnergySystem.getSystem()["intervalBetweenYears"]
    if model.carbonEmissionsBudget != np.inf:
        return (
                model.carbonEmissionsBudget + model.carbonEmissionsOvershoot[year] >=
                model.carbonEmissionsCumulative[year] + model.carbonEmissionsTotal[year] * (intervalBetweenYears - 1)
        )
    else:
        return pe.Constraint.Skip

def constraintCostTotalRule(model, year):
    """ add up all costs from technologies and carriers"""
    return (
            model.costTotal[year] ==
            # capex
            model.capexTotal[year] +
            # opex
            model.opexTotal[year] +
            # carrier costs
            model.costCarrierTotal[year] +
            # carbon costs
            model.costCarbonEmissionsTotal[year]
    )

def constraintNPVRule(model, year):
    """ discounts the annual capital flows to calculate the NPV """
    system = EnergySystem.getSystem()
    discountRate = EnergySystem.getAnalysis()["discountRate"]
    return (
            model.NPV[year] ==
            model.costTotal[year] *
            # economic discount
            ((1 / (1 + discountRate)) ** (system["intervalBetweenYears"] * (year - model.setTimeStepsYearly.at(1))))
    )

# objective rules
def objectiveTotalCostRule(model):
    """objective function to minimize the total cost"""
    system = EnergySystem.getSystem()
    return (
        sum(
            model.NPV[year] *
            # discounted utility function
            ((1 / (1 + system["socialDiscountRate"])) ** (
                        system["intervalBetweenYears"] * (year - model.setTimeStepsYearly.at(1))))
            for year in model.setTimeStepsYearly)
    )

def objectiveNPVRule(model,year):
    """ objective function to minimize NPV """
    return (
        sum(
            model.NPV[year]
            for year in model.setTimeStepsYearly)
    )

def objectiveTotalCarbonEmissionsRule(model):
    """objective function to minimize total emissions"""
    return (sum(model.carbonEmissionsTotal[year] for year in model.setTimeStepsYearly))

def objectiveRiskRule(model):
    """objective function to minimize total risk"""
    # TODO implement objective functions for risk
    return pe.Constraint.Skip

