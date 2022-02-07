"""===========================================================================================================================================================================
Title:          ENERGY-CARBON OPTIMIZATION PLATFORM
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining a generic energy carrier.
                The class takes as inputs the abstract optimization model. The class adds parameters, variables and
                constraints of a generic carrier and returns the abstract optimization model.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
from model.objects.element import Element
# from preprocess.functions.calculate_input_data import extractInputData
from model.objects.energy_system import EnergySystem

class Carrier(Element):
    # empty list of elements
    listOfElements = []

    def __init__(self,carrier):
        """initialization of a generic carrier object
        :param carrier: carrier that is added to the model"""

        logging.info('initialize object of a generic carrier')
        super().__init__(carrier)
        # store input data
        self.storeInputData()
        # add carrier to list
        Carrier.addElement(self)

    def storeInputData(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """   
        # get paths
        paths                           = EnergySystem.getPaths()   
        # set attributes of carrier
        _inputPath                      = paths["setCarriers"][self.name]["folder"]
        self.setTimeStepsCarrier        = self.dataInput.extractTimeSteps(_inputPath)
        self.timeStepsCarrierDuration   = EnergySystem.calculateTimeStepDuration(self.setTimeStepsCarrier)
        self.demandCarrier              = self.dataInput.extractInputData(_inputPath,"demandCarrier",["setNodes","setTimeSteps"],timeSteps=self.setTimeStepsCarrier)
        self.availabilityCarrier        = self.dataInput.extractInputData(_inputPath,"availabilityCarrier",["setNodes","setTimeSteps"],timeSteps=self.setTimeStepsCarrier)
        self.exportPriceCarrier         = self.dataInput.extractInputData(_inputPath,"exportPriceCarrier",["setNodes","setTimeSteps"],column="exportPriceCarrier",timeSteps=self.setTimeStepsCarrier)
        self.importPriceCarrier         = self.dataInput.extractInputData(_inputPath,"importPriceCarrier",["setNodes","setTimeSteps"],column="importPriceCarrier",timeSteps=self.setTimeStepsCarrier)

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Carrier --- ###
    @classmethod
    def constructSets(cls):
        """ constructs the pe.Sets of the class <Carrier> """
        model = EnergySystem.getConcreteModel()
        # time-steps
        model.setTimeStepsCarrier = pe.Set(
            model.setCarriers,
            initialize=cls.getAttributeOfAllElements("setTimeStepsCarrier"),
            doc='Set of time steps of carriers')
        
    @classmethod
    def constructParams(cls):
        """ constructs the pe.Params of the class <Carrier> """
        model = EnergySystem.getConcreteModel()

        # invest time step duration
        model.timeStepsCarrierDuration = pe.Param(
            cls.createCustomSet(["setCarriers","setTimeStepsCarrier"]),
            initialize = cls.getAttributeOfAllElements("timeStepsCarrierDuration"),
            doc="Parameter which specifies the time step duration for all carriers. Dimensions: setCarriers, setTimeStepsCarrier"
        )
        # demand of carrier
        model.demandCarrier = pe.Param(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            initialize = cls.getAttributeOfAllElements("demandCarrier"),
            doc = 'Parameter which specifies the carrier demand.\n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier')
        # availability of carrier
        model.availabilityCarrier = pe.Param(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            initialize = cls.getAttributeOfAllElements("availabilityCarrier"),
            doc = 'Parameter which specifies the maximum energy that can be imported from the grid. \n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier')
        # import price
        model.importPriceCarrier = pe.Param(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            initialize = cls.getAttributeOfAllElements("importPriceCarrier"),
            doc = 'Parameter which specifies the import carrier price. \n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier'
        )
        # export price
        model.exportPriceCarrier = pe.Param(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            initialize = cls.getAttributeOfAllElements("exportPriceCarrier"),
            doc = 'Parameter which specifies the export carrier price. \n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier'
        )

    @classmethod
    def constructVars(cls):
        """ constructs the pe.Vars of the class <Carrier> """
        model = EnergySystem.getConcreteModel()
        
        # flow of imported carrier
        model.importCarrierFlow = pe.Var(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            domain = pe.NonNegativeReals,
            doc = 'node- and time-dependent carrier import from the grid. \n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier. Domain: NonNegativeReals'
        )
        # flow of exported carrier
        model.exportCarrierFlow = pe.Var(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            domain = pe.NonNegativeReals,
            doc = 'node- and time-dependent carrier export from the grid. \n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier. Domain: NonNegativeReals'
        )

    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <Carrier> """
        model = EnergySystem.getConcreteModel()

        # limit import flow by availability
        model.constraintAvailabilityCarrier = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            rule = constraintAvailabilityCarrierRule,
            doc = 'node- and time-dependent carrier availability. \n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier',
        )        
        ### TODO add mass balance but move after technologies
        # energy balance
        model.constraintNodalEnergyBalance = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            rule = constraintNodalEnergyBalanceRule,
            doc = 'node- and time-dependent energy balance for each carrier. \n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier',
        )

#%% Constraint rules defined in current class
def constraintAvailabilityCarrierRule(model, carrier, node, time):
    """node- and time-dependent carrier availability"""

    return(model.importCarrierFlow[carrier, node, time] <= model.availabilityCarrier[carrier,node,time])

# energy balance
def constraintNodalEnergyBalanceRule(model, carrier, node, time):
    """" 
    nodal energy balance for each time step
    """
    # decode to baseTimeStep
    baseTimeStep = EnergySystem.decodeTimeStep(carrier,time)
    # carrier input and output conversion technologies
    carrierConversionIn, carrierConversionOut = 0, 0
    if hasattr(model, 'setConversionTechnologies'):
        for tech in model.setConversionTechnologies:
            operationTimeStep = EnergySystem.encodeTimeStep(tech,baseTimeStep,"operation")
            if carrier in model.setInputCarriers[tech]:
                carrierConversionIn += model.inputFlow[tech,carrier,node,operationTimeStep]
            if carrier in model.setOutputCarriers[tech]:
                carrierConversionOut += model.outputFlow[tech,carrier,node,operationTimeStep]
    # carrier flow transport technologies
    carrierFlowIn, carrierFlowOut = 0, 0
    if hasattr(model, 'setTransportTechnologies'):
        for tech in model.setTransportTechnologies:
            operationTimeStep = EnergySystem.encodeTimeStep(tech,baseTimeStep,"operation")
            if carrier in model.setTransportCarriers[tech]:
                carrierFlowIn += sum(model.carrierFlow[tech,carrier, edge, operationTimeStep]
                                    - model.carrierLoss[tech,carrier, edge, operationTimeStep] for edge in model.setEdges if node == model.setNodesOnEdges[edge][2]) # second entry is node into which the flow goes
                carrierFlowOut += sum(model.carrierFlow[tech,carrier, edge, operationTimeStep] for edge in model.setEdges if node == model.setNodesOnEdges[edge][1]) # first entry is node out of which the flow starts
    # carrier import, demand and export
    carrierImport, carrierExport, carrierDemand = 0, 0, 0
    carrierImport = model.importCarrierFlow[carrier, node, time]
    carrierDemand = model.demandCarrier[carrier, node, time]
    carrierExport = model.exportCarrierFlow[carrier, node, time]

    # TODO implement storage

    return (
        carrierConversionOut - carrierConversionIn + carrierFlowIn - carrierFlowOut + carrierImport - carrierExport - carrierDemand == 0
        )