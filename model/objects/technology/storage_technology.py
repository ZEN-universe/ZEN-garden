"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining the parameters, variables and constraints that hold for all storage technologies.
                The class takes the abstract optimization model as an input, and returns the parameters, variables and
                constraints that hold for the storage technologies.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
import numpy as np
from model.objects.technology.technology import Technology
from model.objects.energy_system import EnergySystem

class StorageTechnology(Technology):
    # set label
    label           = "setStorageTechnologies"
    locationType    = "setNodes"
    # empty list of elements
    listOfElements = []
    
    def __init__(self, tech):
        """init storage technology object
        :param tech: name of added technology"""

        logging.info(f'Initialize storage technology {tech}')
        super().__init__(tech)
        # store input data
        self.storeInputData()
        # add StorageTechnology to list
        StorageTechnology.addElement(self)

    def storeInputData(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """   
        # get attributes from class <Technology>
        super().storeInputData()
        setBaseTimeStepsYearly = EnergySystem.getEnergySystem().setBaseTimeStepsYearly
        # set attributes for parameters of child class <StorageTechnology>
        self.efficiencyCharge               = self.dataInput.extractInputData("efficiencyCharge",indexSets=["setNodes"])
        self.efficiencyDischarge            = self.dataInput.extractInputData("efficiencyDischarge",indexSets=["setNodes"])
        self.selfDischarge                  = self.dataInput.extractInputData("selfDischarge",indexSets=["setNodes"])
        # extract existing energy capacity
        self.minBuiltCapacityEnergy         = self.dataInput.extractAttributeData("minBuiltCapacityEnergy")["value"]
        self.maxBuiltCapacityEnergy         = self.dataInput.extractAttributeData("maxBuiltCapacityEnergy")["value"]
        self.capacityLimitEnergy            = self.dataInput.extractInputData("capacityLimitEnergy",indexSets=["setNodes"])
        # self.setExistingTechnologiesEnergy  = self.dataInput.extractSetExistingTechnologies(storageEnergy = True)
        self.existingCapacityEnergy         = self.dataInput.extractInputData(
            "existingCapacityEnergy",indexSets=["setNodes","setExistingTechnologies"],column="existingCapacity")
        # self.lifetimeExistingTechnologyEnergy = self.dataInput.extractLifetimeExistingTechnology(
        #     "existingCapacityEnergy",indexSets=["setNodes","setExistingTechnologiesEnergy"])
        self.capexSpecific                  = self.dataInput.extractInputData(
            "capexSpecific",indexSets=["setNodes","setTimeSteps"],timeSteps= self.setTimeStepsInvest)
        self.capexSpecificEnergy            = self.dataInput.extractInputData(
            "capexSpecificEnergy",indexSets=["setNodes","setTimeSteps"],timeSteps=self.setTimeStepsInvest)
        # annualize capex
        self.convertToAnnualizedCapex()
        # calculate capex of existing capacity
        self.capexExistingCapacity          = self.calculateCapexOfExistingCapacities()
        self.capexExistingCapacityEnergy    = self.calculateCapexOfExistingCapacities(storageEnergy = True)
        # add min load max load time series for energy
        self.rawTimeSeries["minLoadEnergy"] = self.dataInput.extractInputData(
            "minLoadEnergy", indexSets=["setNodes", "setTimeSteps"],timeSteps=setBaseTimeStepsYearly)
        self.rawTimeSeries["maxLoadEnergy"] = self.dataInput.extractInputData(
            "maxLoadEnergy",indexSets=["setNodes", "setTimeSteps"],timeSteps=setBaseTimeStepsYearly)

    def convertToAnnualizedCapex(self):
        """ this method converts the total capex to annualized capex """
        fractionalAnnuity           = self.calculateFractionalAnnuity()
        # annualize capex
        self.capexSpecific          = self.capexSpecific        * fractionalAnnuity
        self.capexSpecificEnergy    = self.capexSpecificEnergy  * fractionalAnnuity

    def calculateCapexOfSingleCapacity(self,capacity,index,storageEnergy = False):
        """ this method calculates the annualized capex of a single existing capacity. """
        if storageEnergy:
            _absoluteCapex = self.capexSpecificEnergy[index] * capacity
        else:
            _absoluteCapex = self.capexSpecific[index] * capacity
        return _absoluteCapex

    def calculateTimeStepsStorageLevel(self):
        """ this method calculates the number of time steps on the storage level, and the sequence in which the storage levels are connected """
        sequenceTimeSteps                   = self.sequenceTimeSteps
        # calculate connected storage levels, i.e., time steps that are constant for 
        IdxLastConnectedStorageLevel        = np.append(np.flatnonzero(np.diff(sequenceTimeSteps)),len(sequenceTimeSteps)-1)
        # empty setTimeStep
        self.setTimeStepsStorageLevel       = []
        self.timeStepsStorageLevelDuration  = {}
        self.sequenceTimeStepsStorageLevel  = np.zeros(np.size(sequenceTimeSteps)).astype(int)
        counterTimeStep                     = 0
        for idxTimeStep,idxStorageLevel in enumerate(IdxLastConnectedStorageLevel):
            self.setTimeStepsStorageLevel.append(idxTimeStep)
            self.timeStepsStorageLevelDuration[idxTimeStep] = len(range(counterTimeStep,idxStorageLevel+1))
            self.sequenceTimeStepsStorageLevel[counterTimeStep:idxStorageLevel+1] = idxTimeStep
            counterTimeStep                 = idxStorageLevel + 1 
        # add sequence to energy system
        EnergySystem.setSequenceTimeSteps(self.name+"StorageLevel",self.sequenceTimeStepsStorageLevel)

    def overwriteTimeSteps(self,baseTimeSteps):
        """ overwrites setTimeStepsStorageLevel """
        super().overwriteTimeSteps(baseTimeSteps)
        setTimeStepsStorageLevel = EnergySystem.encodeTimeStep(self.name+"StorageLevel", baseTimeSteps=baseTimeSteps,timeStepType="operation", yearly=True)
        setattr(self, "setTimeStepsStorageLevel", setTimeStepsStorageLevel.squeeze().tolist())

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to StorageTechnology --- ###
    @classmethod
    def constructSets(cls):
        """ constructs the pe.Sets of the class <StorageTechnology> """
        model = EnergySystem.getConcreteModel()
        # time steps of storage levels
        model.setTimeStepsStorageLevel = pe.Set(
            model.setStorageTechnologies,
            initialize = cls.getAttributeOfAllElements("setTimeStepsStorageLevel"),
            doc="Set of time steps of storage levels for all storage technologies. Dimensions: setStorageTechnologies"
        )

    @classmethod
    def constructParams(cls):
        """ constructs the pe.Params of the class <StorageTechnology> """
        model = EnergySystem.getConcreteModel()
        
        # time step duration of storage level
        model.timeStepsStorageLevelDuration = pe.Param(
            cls.createCustomSet(["setStorageTechnologies","setTimeStepsStorageLevel"]),
            initialize = EnergySystem.initializeComponent(cls,"timeStepsStorageLevelDuration",indexNames=["setStorageTechnologies","setTimeStepsStorageLevel"]).astype(int),
            doc="Parameter which specifies the time step duration in StorageLevel for all technologies. Dimensions: setStorageTechnologies, setTimeStepsStorageLevel"
        )
        # efficiency charge
        model.efficiencyCharge = pe.Param(
            cls.createCustomSet(["setStorageTechnologies","setNodes"]),
            initialize = EnergySystem.initializeComponent(cls,"efficiencyCharge"),
            doc = 'efficiency during charging for storage technologies. Dimensions: setStorageTechnologies, setNodes'
        )
        # efficiency discharge
        model.efficiencyDischarge = pe.Param(
            cls.createCustomSet(["setStorageTechnologies","setNodes"]),
            initialize = EnergySystem.initializeComponent(cls,"efficiencyDischarge"),
            doc = 'efficiency during discharging for storage technologies. Dimensions: setStorageTechnologies, setNodes'
        )
        # self discharge
        model.selfDischarge = pe.Param(
            cls.createCustomSet(["setStorageTechnologies","setNodes"]),
            initialize = EnergySystem.initializeComponent(cls,"selfDischarge"),
            doc = 'self discharge of storage technologies. Dimensions: setStorageTechnologies, setNodes'
        )
        # capex specific
        model.capexSpecificStorage = pe.Param(
            cls.createCustomSet(["setStorageTechnologies","setCapacityTypes","setNodes","setTimeStepsInvest"]),
            initialize = EnergySystem.initializeComponent(cls,"capexSpecific",indexNames=["setStorageTechnologies","setCapacityTypes","setNodes","setTimeStepsInvest"],capacityTypes=True),
            doc = 'specific capex of storage technologies. Dimensions: setStorageTechnologies, setNodes, setTimeStepsInvest'
        )

    @classmethod
    def constructVars(cls):
        """ constructs the pe.Vars of the class <StorageTechnology> """

        def carrierFlowBounds(model,tech ,node,time):
            """ return bounds of carrierFlow for bigM expression 
            :param model: pe.ConcreteModel
            :param tech: tech index
            :param node: node index
            :param time: time index
            :return bounds: bounds of carrierFlow"""
            # convert operationTimeStep to investTimeStep: operationTimeStep -> baseTimeStep -> investTimeStep
            investTimeStep = EnergySystem.convertTechnologyTimeStepType(tech,time,"operation2invest")
            bounds = model.capacity[tech,"power",node,investTimeStep].bounds
            return bounds

        model = EnergySystem.getConcreteModel()
        # flow of carrier on node into storage
        model.carrierFlowCharge = pe.Var(
            cls.createCustomSet(["setStorageTechnologies","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            bounds = carrierFlowBounds,
            doc = 'carrier flow into storage technology on node i and time t. Dimensions: setStorageTechnologies, setNodes, setTimeStepsOperation. Domain: NonNegativeReals'
        )
        # flow of carrier on node out of storage
        model.carrierFlowDischarge = pe.Var(
            cls.createCustomSet(["setStorageTechnologies","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            bounds = carrierFlowBounds,
            doc = 'carrier flow out of storage technology on node i and time t. Dimensions: setStorageTechnologies, setNodes, setTimeStepsOperation. Domain: NonNegativeReals'
        )
        # loss of carrier on node
        model.levelCharge = pe.Var(
            cls.createCustomSet(["setStorageTechnologies","setNodes","setTimeStepsStorageLevel"]),
            domain = pe.NonNegativeReals,
            doc = 'carrier flow through storage technology on node i and time t. Dimensions: setStorageTechnologies, setNodes, setTimeStepsStorageLevel. Domain: NonNegativeReals'
        )
        
    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <StorageTechnology> """
        model = EnergySystem.getConcreteModel()
        # Limit storage level
        model.constraintStorageLevelMax = pe.Constraint(
            cls.createCustomSet(["setStorageTechnologies","setNodes","setTimeStepsStorageLevel"]),
            rule = constraintStorageLevelMaxRule,
            doc = 'limit maximum storage level to capacity. Dimensions: setStorageTechnologies, setNodes, setTimeStepsStorageLevel'
        ) 
        # couple storage levels
        model.constraintCoupleStorageLevel = pe.Constraint(
            cls.createCustomSet(["setStorageTechnologies","setNodes","setTimeStepsStorageLevel"]), #setTimeStepsStorageLevel setBaseTimeSteps
            rule = constraintCoupleStorageLevelRule,
            doc = 'couple subsequent storage levels (time coupling constraints). Dimensions: setStorageTechnologies, setNodes, setTimeStepsStorageLevel'
        )
        # Linear Capex
        model.constraintStorageTechnologyLinearCapex = pe.Constraint(
            cls.createCustomSet(["setStorageTechnologies","setCapacityTypes","setNodes","setTimeStepsInvest"]),
            rule = constraintCapexStorageTechnologyRule,
            doc = 'Capital expenditures for installing storage technology. Dimensions: setStorageTechnologies,"setCapacityTypes", setNodes, setTimeStepsInvest'
        ) 

    # defines disjuncts if technology on/off
    @classmethod
    def disjunctOnTechnologyRule(cls,disjunct, tech, node, time):
        """definition of disjunct constraints if technology is on"""
        model = disjunct.model()
        # get invest time step
        baseTimeStep = EnergySystem.decodeTimeStep(tech,time,"operation")
        investTimeStep = EnergySystem.encodeTimeStep(tech,baseTimeStep,"invest")
        # disjunct constraints min load charge
        disjunct.constraintMinLoadCharge = pe.Constraint(
            expr=model.carrierFlowCharge[tech, node, time] >= model.minLoad[tech,node,time] * model.capacity[tech,node, investTimeStep]
        )
        # disjunct constraints min load discharge
        disjunct.constraintMinLoadDischarge = pe.Constraint(
            expr=model.carrierFlowDischarge[tech, node, time] >= model.minLoad[tech,node,time] * model.capacity[tech,node, investTimeStep]
        )

    @classmethod
    def disjunctOffTechnologyRule(cls,disjunct, tech, node, time):
        """definition of disjunct constraints if technology is off"""
        model = disjunct.model()
        # off charging
        disjunct.constraintNoLoadCharge = pe.Constraint(
            expr=model.carrierFlowCharge[tech, node, time] == 0
        )
        # off discharging
        disjunct.constraintNoLoadDischarge = pe.Constraint(
            expr=model.carrierFlowDischarge[tech, node, time] == 0
        )

    @classmethod
    def getStorageLevelTimeStep(cls,tech,time):
        """ gets current and previous time step of storage level """
        sequenceStorageLevel    = cls.getAttributeOfSpecificElement(tech,"sequenceStorageLevel")
        setTimeStepsOperation   = cls.getAttributeOfSpecificElement(tech,"setTimeStepsOperation")
        indexCurrentTimeStep    = setTimeStepsOperation.index(time)
        currentLevelTimeStep    = sequenceStorageLevel[indexCurrentTimeStep]
        # if first time step
        if indexCurrentTimeStep == 0:
            previousLevelTimeStep = sequenceStorageLevel[-1]
        # if any other time step
        else:
            previousLevelTimeStep = sequenceStorageLevel[indexCurrentTimeStep-1]
        return currentLevelTimeStep,previousLevelTimeStep

### --- functions with constraint rules --- ###
def constraintStorageLevelMaxRule(model, tech, node, time):
    """limit maximum storage level to capacity"""
    # get invest time step
    baseTimeStep    = EnergySystem.decodeTimeStep(tech+"StorageLevel",time)
    elementTimeStep = EnergySystem.encodeTimeStep(tech,baseTimeStep)
    investTimeStep  = EnergySystem.convertTechnologyTimeStepType(tech,elementTimeStep,"operation2invest")
    return(model.levelCharge[tech, node, time] <= model.capacity[tech,"energy", node, investTimeStep])

def constraintCoupleStorageLevelRule(model, tech, node, time):
    """couple subsequent storage levels (time coupling constraints)"""
    baseTimeStep                = EnergySystem.decodeTimeStep(tech+"StorageLevel",time)
    elementTimeStep             = EnergySystem.encodeTimeStep(tech,baseTimeStep)
    currentLevelTimeStep        = time
    if time != model.setTimeStepsStorageLevel[tech].at(1):
        previousLevelTimeStep   = time-1
    else:
        previousLevelTimeStep   = model.setTimeStepsStorageLevel[tech].at(-1)
    return(
        model.levelCharge[tech, node, currentLevelTimeStep] == 
        model.levelCharge[tech, node, previousLevelTimeStep]*(1-model.selfDischarge[tech,node])**model.timeStepsStorageLevelDuration[tech,time] + 
        (model.carrierFlowCharge[tech, node, elementTimeStep]*model.efficiencyCharge[tech,node] - 
        model.carrierFlowDischarge[tech, node, elementTimeStep]/model.efficiencyDischarge[tech,node])*sum((1-model.selfDischarge[tech,node])**interimTimeStep for interimTimeStep in range(0,model.timeStepsStorageLevelDuration[tech,time]))
    )

def constraintCapexStorageTechnologyRule(model, tech,capacityType, node, time):
    """ definition of the capital expenditures for the storage technology"""
    return (model.capex[tech,capacityType,node, time] ==
            model.builtCapacity[tech,capacityType,node, time] *
            model.capexSpecificStorage[tech,capacityType,node, time])


