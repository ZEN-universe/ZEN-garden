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
        self.availabilityCarrierImport  = self.dataInput.extractInputData(_inputPath,"availabilityCarrier",["setNodes","setTimeSteps"],column="availabilityCarrierImport",timeSteps=self.setTimeStepsCarrier)
        self.availabilityCarrierExport  = self.dataInput.extractInputData(_inputPath,"availabilityCarrier",["setNodes","setTimeSteps"],column="availabilityCarrierExport",timeSteps=self.setTimeStepsCarrier)
        self.exportPriceCarrier         = self.dataInput.extractInputData(_inputPath,"priceCarrier",["setNodes","setTimeSteps"],column="exportPriceCarrier",timeSteps=self.setTimeStepsCarrier)
        self.importPriceCarrier         = self.dataInput.extractInputData(_inputPath,"priceCarrier",["setNodes","setTimeSteps"],column="importPriceCarrier",timeSteps=self.setTimeStepsCarrier)
        self.carbonIntensityCarrier     = self.dataInput.extractInputData(_inputPath,"carbonIntensity",["setNodes"])

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
        model.availabilityCarrierImport = pe.Param(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            initialize = cls.getAttributeOfAllElements("availabilityCarrierImport"),
            doc = 'Parameter which specifies the maximum energy that can be imported from outside the system boundaries. \n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier')
        # availability of carrier
        model.availabilityCarrierExport = pe.Param(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            initialize = cls.getAttributeOfAllElements("availabilityCarrierExport"),
            doc = 'Parameter which specifies the maximum energy that can be exported to outside the system boundaries. \n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier')
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
        # carbon intensity
        model.carbonIntensityCarrier = pe.Param(
            cls.createCustomSet(["setCarriers","setNodes"]),
            initialize = cls.getAttributeOfAllElements("carbonIntensityCarrier"),
            doc = 'Parameter which specifies the carbon intensity of carrier. \n\t Dimensions: setCarriers, setNodes'
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
        # carrier import/export cost
        model.costCarrier = pe.Var(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            domain = pe.NonNegativeReals,
            doc = 'node- and time-dependent carrier cost due to import and export. \n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier. Domain: NonNegativeReals'
        )
        # total carrier import/export cost
        model.costCarrierTotal = pe.Var(
            domain = pe.NonNegativeReals,
            doc = 'total carrier cost due to import and export. \n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier. Domain: NonNegativeReals'
        )
        # carbon emissions
        model.carbonEmissionsCarrier = pe.Var(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            domain = pe.NonNegativeReals,
            doc = "carbon emissions of importing/exporting carrier. Dimensions: setCarriers, setNodes, setTimeStepsCarrier. Domain: NonNegativeReals"
        )
        # total carbon emissions
        model.carbonEmissionsCarrierTotal = pe.Var(
            domain = pe.NonNegativeReals,
            doc = "total carbon emissions of importing/exporting carrier. Domain: NonNegativeReals"
        )

    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <Carrier> """
        model = EnergySystem.getConcreteModel()

        # limit import flow by availability
        model.constraintAvailabilityCarrierImport = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            rule = constraintAvailabilityCarrierImportRule,
            doc = 'node- and time-dependent carrier availability to import from outside the system boundaries. \n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier',
        )        
        # limit export flow by availability
        model.constraintAvailabilityCarrierExport = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            rule = constraintAvailabilityCarrierExportRule,
            doc = 'node- and time-dependent carrier availability to export to outside the system boundaries. \n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier',
        )        
        # cost for carrier 
        model.constraintCostCarrier = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            rule = constraintCostCarrierRule,
            doc = "cost of importing/exporting carrier. Dimensions: setCarriers, setNodes, setTimeStepsCarrier."
        )
        # total cost for carriers
        model.constraintCostCarrierTotal = pe.Constraint(
            rule = constraintCostCarrierTotalRule,
            doc = "total cost of importing/exporting carriers. ."
        )
        # carbon emissions
        model.constraintCarbonEmissionsCarrier = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            rule = constraintCarbonEmissionsCarrierRule,
            doc = "carbon emissions of importing/exporting carrier. Dimensions: setCarriers, setNodes, setTimeStepsCarrier."
        )
        # total carbon emissions
        model.constraintCarbonEmissionsCarrierTotal = pe.Constraint(
            rule = constraintCarbonEmissionsCarrierTotalRule,
            doc = "total carbon emissions of importing/exporting carriers."
        )
        # energy balance
        model.constraintNodalEnergyBalance = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"]),
            rule = constraintNodalEnergyBalanceRule,
            doc = 'node- and time-dependent energy balance for each carrier. \n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier',
        )


#%% Constraint rules defined in current class
def constraintAvailabilityCarrierImportRule(model, carrier, node, time):
    """node- and time-dependent carrier availability to import from outside the system boundaries"""

    return(model.importCarrierFlow[carrier, node, time] <= model.availabilityCarrierImport[carrier,node,time])

def constraintAvailabilityCarrierExportRule(model, carrier, node, time):
    """node- and time-dependent carrier availability to export to outside the system boundaries"""

    return(model.exportCarrierFlow[carrier, node, time] <= model.availabilityCarrierExport[carrier,node,time])

def constraintCostCarrierRule(model, carrier, node, time):
    """ carbon emissions of importing/exporting carrier"""
    return(model.costCarrier[carrier,node, time] == 
        model.importPriceCarrier[carrier, node, time]*model.importCarrierFlow[carrier, node, time] - 
        model.exportPriceCarrier[carrier, node, time]*model.exportCarrierFlow[carrier, node, time]
    )

def constraintCostCarrierTotalRule(model):
    """ total carbon emissions of importing/exporting carrier"""
    return(model.costCarrierTotal == 
        sum(
            model.costCarrier[carrier,node,time]*model.timeStepsCarrierDuration[carrier, time]
            for carrier,node,time in Element.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"])
        )
    )

def constraintCarbonEmissionsCarrierRule(model, carrier, node, time):
    """ carbon emissions of importing/exporting carrier"""

    return(model.carbonEmissionsCarrier[carrier,node, time] == 
        model.carbonIntensityCarrier[carrier,node]*
        (model.importCarrierFlow[carrier, node, time] - model.exportCarrierFlow[carrier, node, time])
    )

def constraintCarbonEmissionsCarrierTotalRule(model):
    """ total carbon emissions of importing/exporting carrier"""

    return(model.carbonEmissionsCarrierTotal == 
        sum(
            model.carbonEmissionsCarrier[carrier,node,time]*model.timeStepsCarrierDuration[carrier, time]
            for carrier,node,time in Element.createCustomSet(["setCarriers","setNodes","setTimeStepsCarrier"])
        )
    )
    

def constraintNodalEnergyBalanceRule(model, carrier, node, time):
    """" 
    nodal energy balance for each time step
    """
    # decode to baseTimeStep
    baseTimeStep = EnergySystem.decodeTimeStep(carrier,time)
    # carrier input and output conversion technologies
    carrierConversionIn, carrierConversionOut = 0, 0
    for tech in model.setConversionTechnologies:
        operationTimeStep = EnergySystem.encodeTimeStep(tech,baseTimeStep,"operation")
        if carrier in model.setInputCarriers[tech]:
            carrierConversionIn += model.inputFlow[tech,carrier,node,operationTimeStep]
        if carrier in model.setOutputCarriers[tech]:
            carrierConversionOut += model.outputFlow[tech,carrier,node,operationTimeStep]
    # carrier flow transport technologies
    carrierFlowIn, carrierFlowOut = 0, 0
    setEdgesIn = EnergySystem.calculateConnectedEdges(node,"in")
    setEdgesOut = EnergySystem.calculateConnectedEdges(node,"out")
    for tech in model.setTransportTechnologies:
        operationTimeStep = EnergySystem.encodeTimeStep(tech,baseTimeStep,"operation")
        if carrier in model.setReferenceCarriers[tech]:
            carrierFlowIn   += sum(model.carrierFlow[tech,carrier, edge, operationTimeStep]
                            - model.carrierLoss[tech,carrier, edge, operationTimeStep] for edge in setEdgesIn) 
            carrierFlowOut  += sum(model.carrierFlow[tech,carrier, edge, operationTimeStep] for edge in setEdgesOut) 
    # carrier import, demand and export
    carrierImport, carrierExport, carrierDemand = 0, 0, 0
    carrierImport = model.importCarrierFlow[carrier, node, time]
    carrierDemand = model.demandCarrier[carrier, node, time]
    carrierExport = model.exportCarrierFlow[carrier, node, time]

    # TODO implement storage

    return (
        carrierConversionOut - carrierConversionIn + carrierFlowIn - carrierFlowOut + carrierImport - carrierExport - carrierDemand == 0
        )