"""===========================================================================================================================================================================
Title:          ENERGY-CARBON OPTIMIZATION PLATFORM
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
from model.objects.technology.technology import Technology
from model.objects.energy_system import EnergySystem

class StorageTechnology(Technology):
    # empty list of elements
    listOfElements = []
    
    def __init__(self, tech):
        """init generic technology object
        :param object: object of the abstract optimization model"""

        logging.info('initialize object of a storage technology')
        super().__init__(tech)
        # store input data
        self.storeInputData()
        # add StorageTechnology to list
        StorageTechnology.addElement(self)

    def storeInputData(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """   
        # get attributes from class <Technology>
        super().storeInputData()
        # get system information
        paths                           = EnergySystem.getPaths()   
        # set attributes for parameters of parent class <Technology>
        _inputPath                      = paths["setStorageTechnologies"][self.name]["folder"]
        self.capacityLimit              = self.dataInput.extractInputData(_inputPath,"capacityLimit",indexSets=["setNodes","setTimeSteps"],timeSteps=self.setTimeStepsInvest)
        self.minLoad                    = self.dataInput.extractInputData(_inputPath,"minLoad",indexSets=["setNodes","setTimeSteps"],timeSteps=self.setTimeStepsOperation)
        self.maxLoad                    = self.dataInput.extractInputData(_inputPath,"maxLoad",indexSets=["setNodes","setTimeSteps"],timeSteps=self.setTimeStepsOperation)
        self.opexSpecific               = self.dataInput.extractInputData(_inputPath,"opexSpecific",indexSets=["setNodes","setTimeSteps"],timeSteps= self.setTimeStepsOperation)
        self.carbonIntensityTechnology  = self.dataInput.extractInputData(_inputPath,"carbonIntensity",indexSets=["setNodes"])
        # set attributes for parameters of child class <StorageTechnology>
        self.efficiencyCharge           = self.dataInput.extractInputData(_inputPath,"efficiencyCharge",indexSets=["setNodes"])
        self.efficiencyDischarge        = self.dataInput.extractInputData(_inputPath,"efficiencyDischarge",indexSets=["setNodes"])
        self.capexSpecific              = self.dataInput.extractInputData(_inputPath,"capexSpecific",indexSets=["setNodes","setTimeSteps"],timeSteps= self.setTimeStepsInvest)
        # set technology to correspondent reference carrier
        EnergySystem.setTechnologyOfCarrier(self.name,self.referenceCarrier)

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to StorageTechnology --- ###
    @classmethod
    def constructSets(cls):
        """ constructs the pe.Sets of the class <StorageTechnology> """
        pass

    @classmethod
    def constructParams(cls):
        """ constructs the pe.Params of the class <StorageTechnology> """
        model = EnergySystem.getConcreteModel()
        
        # efficiency charge
        model.efficiencyCharge = pe.Param(
            cls.createCustomSet(["setStorageTechnologies","setNodes"]),
            initialize = cls.getAttributeOfAllElements("efficiencyCharge"),
            doc = 'efficiency during charging for storage technologies. Dimensions: setStorageTechnologies, setNodes'
        )
        # efficiency discharge
        model.efficiencyDischarge = pe.Param(
            cls.createCustomSet(["setStorageTechnologies","setNodes"]),
            initialize = cls.getAttributeOfAllElements("efficiencyDischarge"),
            doc = 'efficiency during discharging for storage technologies. Dimensions: setStorageTechnologies, setNodes'
        )
        # capex specific
        model.capexSpecific = pe.Param(
            cls.createCustomSet(["setStorageTechnologies","setNodes","setTimeStepsInvest"]),
            initialize = cls.getAttributeOfAllElements("capexSpecific"),
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
            baseTimeStep = EnergySystem.decodeTimeStep(tech,time,"operation")
            investTimeStep = EnergySystem.encodeTimeStep(tech,baseTimeStep,"invest")
            bounds = model.capacity[tech,node,investTimeStep].bounds
            return(bounds)

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
            cls.createCustomSet(["setStorageTechnologies","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            doc = 'carrier flow through storage technology on node i and time t. Dimensions: setStorageTechnologies, setNodes, setTimeStepsOperation. Domain: NonNegativeReals'
        )
        
    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <StorageTechnology> """
        model = EnergySystem.getConcreteModel()
        return
        # Carrier Flow Losses 
        model.constraintStorageTechnologyLossesFlow = pe.Constraint(
            cls.createCustomSet(["setStorageTechnologies","setNodes","setTimeStepsOperation"]),
            rule = constraintStorageTechnologyLossesFlowRule,
            doc = 'Carrier loss due to storage with through storage technology. Dimensions: setStorageTechnologies, setNodes, setTimeStepsOperation'
        ) 
        # Linear Capex
        model.constraintStorageTechnologyLinearCapex = pe.Constraint(
            cls.createCustomSet(["setStorageTechnologies","setNodes","setTimeStepsInvest"]),
            rule = constraintCapexStorageTechnologyRule,
            doc = 'Capital expenditures for installing storage technology. Dimensions: setStorageTechnologies, setNodes, setTimeStepsInvest'
        ) 

    # defines disjuncts if technology on/off
    @classmethod
    def disjunctOnTechnologyRule(cls,disjunct, tech, node, time):
        """definition of disjunct constraints if technology is on"""
        return
        model = disjunct.model()
        referenceCarrier = model.setReferenceCarriers[tech][1]
        # get invest time step
        baseTimeStep = EnergySystem.decodeTimeStep(tech,time,"operation")
        investTimeStep = EnergySystem.encodeTimeStep(tech,baseTimeStep,"invest")
        # disjunct constraints min load
        disjunct.constraintMinLoad = pe.Constraint(
            expr=model.carrierFlow[tech,referenceCarrier, node, time] >= model.minLoad[tech,node,time] * model.capacity[tech,node, investTimeStep]
        )

    @classmethod
    def disjunctOffTechnologyRule(cls,disjunct, tech, node, time):
        """definition of disjunct constraints if technology is off"""
        return
        model = disjunct.model()
        referenceCarrier = model.setReferenceCarriers[tech][1]
        disjunct.constraintNoLoad = pe.Constraint(
            expr=model.carrierFlow[tech,referenceCarrier, node, time] == 0
        )

### --- functions with constraint rules --- ###
def constraintStorageTechnologyLossesFlowRule(model, tech, node, time):
    """compute the flow losses for a carrier through a storage technology"""
    referenceCarrier = model.setReferenceCarriers[tech][1]
    return(model.carrierLoss[tech,referenceCarrier, node, time]
            == model.distance[tech,node] * model.lossFlow[tech] * model.carrierFlow[tech,referenceCarrier, node, time])

def constraintCapexStorageTechnologyRule(model, tech, node, time):
    """ definition of the capital expenditures for the storage technology"""
    # TODO: why factor 0.5? divide capexPerDistance in input data
    return (model.capex[tech,node, time] == 0.5 *
            model.builtCapacity[tech,node, time] *
            model.distance[tech,node] *
            model.capexPerDistance[tech,node, time])

def constraintOpexStorageTechnologyRule(model, tech, node, time):
    """ definition of the opex for the storage technology"""
    # TODO: why factor 0.5? divide capexPerDistance in input data
    return (model.opex[tech,node, time] == 0.5 *
            model.builtCapacity[tech,node, time] *
            model.distance[tech,node] *
            model.capexPerDistance[tech,node, time])
