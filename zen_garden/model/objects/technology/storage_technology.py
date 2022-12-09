"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class defining the parameters, variables and constraints that hold for all storage technologies.
                The class takes the abstract optimization model as an input, and returns the parameters, variables and
                constraints that hold for the storage technologies.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
import numpy as np
from .technology import Technology
from ..energy_system import EnergySystem
from ..component import Parameter,Variable,Constraint

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
        set_base_time_steps_yearly  = EnergySystem.get_energy_system().set_base_time_steps_yearly
        set_time_steps_yearly      = EnergySystem.get_energy_system().set_time_steps_yearly
        # set attributes for parameters of child class <StorageTechnology>
        self.efficiencyCharge               = self.datainput.extract_input_data("efficiencyCharge",index_sets=["setNodes","setTimeSteps"],time_steps= set_time_steps_yearly)
        self.efficiencyDischarge            = self.datainput.extract_input_data("efficiencyDischarge",index_sets=["setNodes","setTimeSteps"],time_steps= set_time_steps_yearly)
        self.selfDischarge                  = self.datainput.extract_input_data("selfDischarge",index_sets=["setNodes"])
        # extract existing energy capacity
        self.minBuiltCapacityEnergy         = self.datainput.extractAttributeData("minBuiltCapacityEnergy")["value"]
        self.maxBuiltCapacityEnergy         = self.datainput.extractAttributeData("maxBuiltCapacityEnergy")["value"]
        self.capacityLimitEnergy            = self.datainput.extract_input_data("capacityLimitEnergy",index_sets=["setNodes"])
        self.existingCapacityEnergy         = self.datainput.extract_input_data("existingCapacityEnergy",index_sets=["setNodes","setExistingTechnologies"],column="existingCapacityEnergy")
        self.existingInvestedCapacityEnergy = self.datainput.extract_input_data("existingInvestedCapacityEnergy", index_sets=["setNodes", "setTimeSteps"],time_steps=set_time_steps_yearly)
        self.capexSpecific                  = self.datainput.extract_input_data(
            "capexSpecific",index_sets=["setNodes","setTimeSteps"],time_steps= set_time_steps_yearly)
        self.capexSpecificEnergy            = self.datainput.extract_input_data(
            "capexSpecificEnergy",index_sets=["setNodes","setTimeSteps"],time_steps=set_time_steps_yearly)
        self.fixedOpexSpecificEnergy        = self.datainput.extract_input_data("fixedOpexSpecificEnergy", index_sets=["setNodes", "setTimeSteps"], time_steps=set_time_steps_yearly)        # annualize capex
        self.convertToAnnualizedCapex()
        # calculate capex of existing capacity
        self.capexExistingCapacity          = self.calculateCapexOfExistingCapacities()
        self.capexExistingCapacityEnergy    = self.calculateCapexOfExistingCapacities(storageEnergy = True)
        # add min load max load time series for energy
        self.raw_time_series["minLoadEnergy"] = self.datainput.extract_input_data(
            "minLoadEnergy", index_sets=["setNodes", "setTimeSteps"],time_steps=set_base_time_steps_yearly)
        self.raw_time_series["maxLoadEnergy"] = self.datainput.extract_input_data(
            "maxLoadEnergy",index_sets=["setNodes", "setTimeSteps"],time_steps=set_base_time_steps_yearly)

    def convertToAnnualizedCapex(self):
        """ this method converts the total capex to annualized capex """
        fractionalAnnuity           = self.calculateFractionalAnnuity()
        system                      = EnergySystem.getSystem()
        _fractionOfYear             = system["unaggregatedTimeStepsPerYear"] / system["totalHoursPerYear"]
        # annualize capex
        self.capexSpecific          = self.capexSpecific        * fractionalAnnuity + self.fixedOpexSpecific * _fractionOfYear
        self.capexSpecificEnergy    = self.capexSpecificEnergy  * fractionalAnnuity + self.fixedOpexSpecificEnergy * _fractionOfYear

    def calculateCapexOfSingleCapacity(self,capacity,index,storageEnergy = False):
        """ this method calculates the annualized capex of a single existing capacity. """
        if storageEnergy:
            _absoluteCapex = self.capexSpecificEnergy[index[0]].iloc[0] * capacity
        else:
            _absoluteCapex = self.capexSpecific[index[0]].iloc[0] * capacity
        return _absoluteCapex

    def calculateTimeStepsStorageLevel(self,conductedTimeSeriesAggregation):
        """ this method calculates the number of time steps on the storage level, and the sequence in which the storage levels are connected
        conductedTimeSeriesAggregation: boolean if the time series were aggregated. If not, the storage level index is the same as the carrier flow indices """
        sequenceTimeSteps                   = self.sequenceTimeSteps
        # if time series aggregation was conducted
        if conductedTimeSeriesAggregation:
            # calculate connected storage levels, i.e., time steps that are constant for
            IdxLastConnectedStorageLevel        = np.append(np.flatnonzero(np.diff(sequenceTimeSteps)),len(sequenceTimeSteps)-1)
            # empty setTimeStep
            self.setTimeStepsStorageLevel       = []
            self.timeStepsStorageLevelDuration  = {}
            timeStepsEnergy2Power               = {}
            self.sequenceTimeStepsStorageLevel  = np.zeros(np.size(sequenceTimeSteps)).astype(int)
            counterTimeStep                     = 0
            for idxTimeStep,idxStorageLevel in enumerate(IdxLastConnectedStorageLevel):
                self.setTimeStepsStorageLevel.append(idxTimeStep)
                self.timeStepsStorageLevelDuration[idxTimeStep] = len(range(counterTimeStep,idxStorageLevel+1))
                self.sequenceTimeStepsStorageLevel[counterTimeStep:idxStorageLevel+1] = idxTimeStep
                timeStepsEnergy2Power[idxTimeStep]  = sequenceTimeSteps[idxStorageLevel]
                counterTimeStep                 = idxStorageLevel + 1
        else:
            self.setTimeStepsStorageLevel       = self.setTimeStepsOperation
            self.timeStepsStorageLevelDuration  = self.timeStepsOperationDuration
            self.sequenceTimeStepsStorageLevel  = sequenceTimeSteps
            timeStepsEnergy2Power               = {idx: idx for idx in self.setTimeStepsOperation}

        # add sequence to energy system
        EnergySystem.setSequenceTimeSteps(self.name+"StorageLevel",self.sequenceTimeStepsStorageLevel)
        # set the dict timeStepsEnergy2Power
        EnergySystem.setTimeStepsEnergy2Power(self.name, timeStepsEnergy2Power)
        # set the first and last time step of each year
        EnergySystem.setTimeStepsStorageStartEnd(self.name)

    def overwrite_time_steps(self,baseTimeSteps):
        """ overwrites setTimeStepsStorageLevel """
        super().overwrite_time_steps(baseTimeSteps)
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
        Parameter.addParameter(
            name ="timeStepsStorageLevelDuration",
            data = EnergySystem.initializeComponent(cls,"timeStepsStorageLevelDuration",index_names=["setStorageTechnologies","setTimeStepsStorageLevel"]),
            doc  ="Parameter which specifies the time step duration in StorageLevel for all technologies"
        )
        # efficiency charge
        Parameter.addParameter(
            name="efficiencyCharge",
            data= EnergySystem.initializeComponent(cls,"efficiencyCharge",index_names=["setStorageTechnologies","setNodes","set_time_steps_yearly"]),
            doc = 'efficiency during charging for storage technologies'
        )
        # efficiency discharge
        Parameter.addParameter(
            name="efficiencyDischarge",
            data= EnergySystem.initializeComponent(cls,"efficiencyDischarge",index_names=["setStorageTechnologies","setNodes","set_time_steps_yearly"]),
            doc = 'efficiency during discharging for storage technologies'
        )
        # self discharge
        Parameter.addParameter(
            name="selfDischarge",
            data= EnergySystem.initializeComponent(cls,"selfDischarge",index_names=["setStorageTechnologies","setNodes"]),
            doc = 'self discharge of storage technologies'
        )
        # capex specific
        Parameter.addParameter(
            name="capexSpecificStorage",
            data= EnergySystem.initializeComponent(cls,"capexSpecific",index_names=["setStorageTechnologies","setCapacityTypes","setNodes","set_time_steps_yearly"],capacityTypes=True),
            doc = 'specific capex of storage technologies'
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
            # convert operationTimeStep to timeStepYear: operationTimeStep -> baseTimeStep -> timeStepYear
            timeStepYear = EnergySystem.convertTimeStepOperation2Invest(tech,time)
            bounds = model.capacity[tech,"power",node,timeStepYear].bounds
            return bounds

        model = EnergySystem.getConcreteModel()
        # flow of carrier on node into storage
        Variable.addVariable(
            model,
            name="carrierFlowCharge",
            index_sets= cls.createCustomSet(["setStorageTechnologies","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            bounds = carrierFlowBounds,
            doc = 'carrier flow into storage technology on node i and time t'
        )
        # flow of carrier on node out of storage
        Variable.addVariable(
            model,
            name="carrierFlowDischarge",
            index_sets= cls.createCustomSet(["setStorageTechnologies","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            bounds = carrierFlowBounds,
            doc = 'carrier flow out of storage technology on node i and time t'
        )
        # loss of carrier on node
        Variable.addVariable(
            model,
            name="levelCharge",
            index_sets= cls.createCustomSet(["setStorageTechnologies","setNodes","setTimeStepsStorageLevel"]),
            domain = pe.NonNegativeReals,
            doc = 'storage level of storage technology ón node in each storage time step'
        )
        
    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <StorageTechnology> """
        model = EnergySystem.getConcreteModel()
        # Limit storage level
        Constraint.addConstraint(
            model,
            name="constraintStorageLevelMax",
            index_sets= cls.createCustomSet(["setStorageTechnologies","setNodes","setTimeStepsStorageLevel"]),
            rule = constraintStorageLevelMaxRule,
            doc = 'limit maximum storage level to capacity'
        ) 
        # couple storage levels
        Constraint.addConstraint(
            model,
            name="constraintCoupleStorageLevel",
            index_sets= cls.createCustomSet(["setStorageTechnologies","setNodes","setTimeStepsStorageLevel"]),
            rule = constraintCoupleStorageLevelRule,
            doc = 'couple subsequent storage levels (time coupling constraints)'
        )
        # Linear Capex
        Constraint.addConstraint(
            model,
            name="constraintStorageTechnologyLinearCapex",
            index_sets= cls.createCustomSet(["setStorageTechnologies","setCapacityTypes","setNodes","set_time_steps_yearly"]),
            rule = constraintCapexStorageTechnologyRule,
            doc = 'Capital expenditures for installing storage technology'
        ) 

    # defines disjuncts if technology on/off
    @classmethod
    def disjunctOnTechnologyRule(cls,disjunct, tech,capacityType, node, time):
        """definition of disjunct constraints if technology is on"""
        model = disjunct.model()
        params = Parameter.getComponentObject()
        # get invest time step
        baseTimeStep = EnergySystem.decodeTimeStep(tech,time,"operation")
        timeStepYear = EnergySystem.encodeTimeStep(tech,baseTimeStep,"yearly")
        # disjunct constraints min load charge
        disjunct.constraintMinLoadCharge = pe.Constraint(
            expr=model.carrierFlowCharge[tech, node, time] >= params.minLoad[tech,capacityType,node,time] * model.capacity[tech,capacityType,node, timeStepYear]
        )
        # disjunct constraints min load discharge
        disjunct.constraintMinLoadDischarge = pe.Constraint(
            expr=model.carrierFlowDischarge[tech, node, time] >= params.minLoad[tech,capacityType,node,time] * model.capacity[tech,capacityType,node, timeStepYear]
        )

    @classmethod
    def disjunctOffTechnologyRule(cls,disjunct, tech, capacityType, node, time):
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
    elementTimeStep = EnergySystem.convertTimeStepEnergy2Power(tech,time)
    timeStepYear  = EnergySystem.convertTimeStepOperation2Invest(tech,elementTimeStep)
    return(model.levelCharge[tech, node, time] <= model.capacity[tech,"energy", node, timeStepYear])

def constraintCoupleStorageLevelRule(model, tech, node, time):
    """couple subsequent storage levels (time coupling constraints)"""
    # get parameter object
    params = Parameter.getComponentObject()
    elementTimeStep             = EnergySystem.convertTimeStepEnergy2Power(tech,time)
    # get invest time step
    timeStepYear              = EnergySystem.convertTimeStepOperation2Invest(tech,elementTimeStep)
    # get corresponding start time step at beginning of the year, if time is last time step in year
    timeStepEnd                 = EnergySystem.getTimeStepsStorageStartEnd(tech,time)
    if timeStepEnd is not None:
        previousLevelTimeStep   = timeStepEnd
    else:
        previousLevelTimeStep   = time-1

    return(
        model.levelCharge[tech, node, time] ==
        model.levelCharge[tech, node, previousLevelTimeStep]*(1-params.selfDischarge[tech,node])**params.timeStepsStorageLevelDuration[tech,time] +
        (model.carrierFlowCharge[tech, node, elementTimeStep]*params.efficiencyCharge[tech,node,timeStepYear] -
        model.carrierFlowDischarge[tech, node, elementTimeStep]/params.efficiencyDischarge[tech,node,timeStepYear])*sum((1-params.selfDischarge[tech,node])**interimTimeStep for interimTimeStep in range(0,params.timeStepsStorageLevelDuration[tech,time]))
    )

def constraintCapexStorageTechnologyRule(model, tech,capacityType, node, time):
    """ definition of the capital expenditures for the storage technology"""
    # get parameter object
    params = Parameter.getComponentObject()
    return (model.capex[tech,capacityType,node, time] ==
            model.built_capacity[tech,capacityType,node, time] *
            params.capexSpecificStorage[tech,capacityType,node, time])
