"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
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
import numpy as np
import pandas as pd
from model.objects.element import Element
from model.objects.energy_system import EnergySystem
from model.objects.technology.technology import Technology

class Carrier(Element):
    # set label
    label = "setCarriers"
    # empty list of elements
    listOfElements = []

    def __init__(self,carrier):
        """initialization of a generic carrier object
        :param carrier: carrier that is added to the model"""

        logging.info(f'Initialize carrier {carrier}')
        super().__init__(carrier)
        # store input data
        self.storeInputData()
        # add carrier to list
        Carrier.addElement(self)

    def storeInputData(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """   
        setBaseTimeStepsYearly          = EnergySystem.getEnergySystem().setBaseTimeStepsYearly
        # set attributes of carrier
        # raw import
        self.rawTimeSeries                              = {}
        self.rawTimeSeries["demandCarrier"]             = self.dataInput.extractInputData("demandCarrier",["setNodes","setTimeSteps"],timeSteps=setBaseTimeStepsYearly)
        self.rawTimeSeries["availabilityCarrierImport"] = self.dataInput.extractInputData("availabilityCarrier",["setNodes","setTimeSteps"],column="availabilityCarrierImport",timeSteps=setBaseTimeStepsYearly)
        self.rawTimeSeries["availabilityCarrierExport"] = self.dataInput.extractInputData("availabilityCarrier",["setNodes","setTimeSteps"],column="availabilityCarrierExport",timeSteps=setBaseTimeStepsYearly)
        self.rawTimeSeries["exportPriceCarrier"]        = self.dataInput.extractInputData("priceCarrier",["setNodes","setTimeSteps"],column="exportPriceCarrier",timeSteps=setBaseTimeStepsYearly)
        self.rawTimeSeries["importPriceCarrier"]        = self.dataInput.extractInputData("priceCarrier",["setNodes","setTimeSteps"],column="importPriceCarrier",timeSteps=setBaseTimeStepsYearly)
        # non-time series input data
        self.carbonIntensityCarrier                     = self.dataInput.extractInputData("carbonIntensity",["setNodes"])
        
    def overwriteTimeSteps(self,baseTimeSteps):
        """ overwrites setTimeStepsOperation and  setTimeStepsEnergyBalance"""
        setTimeStepsOperation         = EnergySystem.encodeTimeStep(self.name, baseTimeSteps=baseTimeSteps, timeStepType="operation",yearly=True)
        setTimeStepsEnergyBalance   = EnergySystem.encodeTimeStep(self.name+"EnergyBalance", baseTimeSteps=baseTimeSteps,timeStepType="operation", yearly=True)
        setattr(self, "setTimeStepsOperation", setTimeStepsOperation.squeeze().tolist())
        setattr(self, "setTimeStepsEnergyBalance", setTimeStepsEnergyBalance.squeeze().tolist())

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Carrier --- ###
    @classmethod
    def constructSets(cls):
        """ constructs the pe.Sets of the class <Carrier> """
        model = EnergySystem.getConcreteModel()
         # time-steps of energy balance of carrier
        model.setTimeStepsEnergyBalance = pe.Set(
            model.setCarriers,
            initialize=cls.getAttributeOfAllElements("setTimeStepsEnergyBalance"),
            doc='Set of time steps of carriers. Dimensions: setCarriers')

    @classmethod
    def constructParams(cls):
        """ constructs the pe.Params of the class <Carrier> """
        model = EnergySystem.getConcreteModel()

        # demand of carrier
        model.demandCarrier = pe.Param(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            initialize = EnergySystem.initializeComponent(cls,"demandCarrier",indexNames=["setCarriers","setNodes","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the carrier demand.\n\t Dimensions: setCarriers, setNodes, setTimeStepsOperation')
        # availability of carrier
        model.availabilityCarrierImport = pe.Param(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            initialize = EnergySystem.initializeComponent(cls,"availabilityCarrierImport",indexNames=["setCarriers","setNodes","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the maximum energy that can be imported from outside the system boundaries. \n\t Dimensions: setCarriers, setNodes, setTimeStepsOperation')
        # availability of carrier
        model.availabilityCarrierExport = pe.Param(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            initialize = EnergySystem.initializeComponent(cls,"availabilityCarrierExport",indexNames=["setCarriers","setNodes","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the maximum energy that can be exported to outside the system boundaries. \n\t Dimensions: setCarriers, setNodes, setTimeStepsOperation')
        # import price
        model.importPriceCarrier = pe.Param(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            initialize = EnergySystem.initializeComponent(cls,"importPriceCarrier",indexNames=["setCarriers","setNodes","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the import carrier price. \n\t Dimensions: setCarriers, setNodes, setTimeStepsOperation'
        )
        # export price
        model.exportPriceCarrier = pe.Param(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            initialize = EnergySystem.initializeComponent(cls,"exportPriceCarrier",indexNames=["setCarriers","setNodes","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the export carrier price. \n\t Dimensions: setCarriers, setNodes, setTimeStepsOperation'
        )
        # carbon intensity
        model.carbonIntensityCarrier = pe.Param(
            cls.createCustomSet(["setCarriers","setNodes"]),
            initialize = EnergySystem.initializeComponent(cls,"carbonIntensityCarrier"),
            doc = 'Parameter which specifies the carbon intensity of carrier. \n\t Dimensions: setCarriers, setNodes'
        )

    @classmethod
    def constructVars(cls):
        """ constructs the pe.Vars of the class <Carrier> """
        model = EnergySystem.getConcreteModel()
        
        # flow of imported carrier
        model.importCarrierFlow = pe.Var(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            doc = 'node- and time-dependent carrier import from the grid. \n\t Dimensions: setCarriers, setNodes, setTimeStepsOperation. Domain: NonNegativeReals'
        )
        # flow of exported carrier
        model.exportCarrierFlow = pe.Var(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            doc = 'node- and time-dependent carrier export from the grid. \n\t Dimensions: setCarriers, setNodes, setTimeStepsOperation. Domain: NonNegativeReals'
        )
        # carrier import/export cost
        model.costCarrier = pe.Var(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.Reals,
            doc = 'node- and time-dependent carrier cost due to import and export. \n\t Dimensions: setCarriers, setNodes, setTimeStepsOperation. Domain: NonNegativeReals'
        )
        # total carrier import/export cost
        model.costCarrierTotal = pe.Var(
            model.setTimeStepsYearly,
            domain = pe.Reals,
            doc = 'total carrier cost due to import and export. \n\t Dimensions: setCarriers, setNodes, setTimeStepsOperation. Domain: NonNegativeReals'
        )
        # carbon emissions
        model.carbonEmissionsCarrier = pe.Var(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.Reals,
            doc = "carbon emissions of importing/exporting carrier. Dimensions: setCarriers, setNodes, setTimeStepsOperation. Domain: NonNegativeReals"
        )
        # carbon emissions carrier
        model.carbonEmissionsCarrierTotal = pe.Var(
            model.setTimeStepsYearly,
            domain=pe.Reals,
            doc="total carbon emissions of importing/exporting carrier. Domain: NonNegativeReals"
        )

    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <Carrier> """
        model = EnergySystem.getConcreteModel()

        # limit import flow by availability
        model.constraintAvailabilityCarrierImport = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            rule = constraintAvailabilityCarrierImportRule,
            doc = 'node- and time-dependent carrier availability to import from outside the system boundaries. \n\t Dimensions: setCarriers, setNodes, setTimeStepsOperation',
        )        
        # limit export flow by availability
        model.constraintAvailabilityCarrierExport = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            rule = constraintAvailabilityCarrierExportRule,
            doc = 'node- and time-dependent carrier availability to export to outside the system boundaries. \n\t Dimensions: setCarriers, setNodes, setTimeStepsOperation',
        )        
        # cost for carrier 
        model.constraintCostCarrier = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            rule = constraintCostCarrierRule,
            doc = "cost of importing/exporting carrier. Dimensions: setCarriers, setNodes, setTimeStepsOperation."
        )
        # total cost for carriers
        model.constraintCostCarrierTotal = pe.Constraint(
            model.setTimeStepsYearly,
            rule = constraintCostCarrierTotalRule,
            doc = "total cost of importing/exporting carriers. ."
        )
        # carbon emissions
        model.constraintCarbonEmissionsCarrier = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            rule = constraintCarbonEmissionsCarrierRule,
            doc = "carbon emissions of importing/exporting carrier. Dimensions: setCarriers, setNodes, setTimeStepsOperation."
        )
        # energy balance
        model.constraintNodalEnergyBalance = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsEnergyBalance"]),
            rule = constraintNodalEnergyBalanceRule,
            doc = 'node- and time-dependent energy balance for each carrier. \n\t Dimensions: setCarriers, setNodes, setTimeStepsEnergyBalance',
        )
        # carbon emissions carrier
        model.constraintCarbonEmissionsCarrierTotal = pe.Constraint(
            model.setTimeStepsYearly,
            rule=constraintCarbonEmissionsCarrierTotalRule,
            doc="total carbon emissions of importing/exporting carriers."
        )


#%% Constraint rules defined in current class
def constraintAvailabilityCarrierImportRule(model, carrier, node, time):
    """node- and time-dependent carrier availability to import from outside the system boundaries"""
    return(model.importCarrierFlow[carrier, node, time] <= model.availabilityCarrierImport[carrier,node,time])

def constraintAvailabilityCarrierExportRule(model, carrier, node, time):
    """node- and time-dependent carrier availability to export to outside the system boundaries"""
    return(model.exportCarrierFlow[carrier, node, time] <= model.availabilityCarrierExport[carrier,node,time])

def constraintCostCarrierRule(model, carrier, node, time):
    """ cost of importing/exporting carrier"""
    return(model.costCarrier[carrier,node, time] == 
        model.importPriceCarrier[carrier, node, time]*model.importCarrierFlow[carrier, node, time] - 
        model.exportPriceCarrier[carrier, node, time]*model.exportCarrierFlow[carrier, node, time]
    )

def constraintCostCarrierTotalRule(model,year):
    """ total cost of importing/exporting carrier"""
    baseTimeStep = EnergySystem.decodeTimeStep(None, year, "yearly")
    return(model.costCarrierTotal[year] ==
        sum(
            sum(
                model.costCarrier[carrier,node,time]*model.timeStepsOperationDuration[carrier, time]
                for time in EnergySystem.encodeTimeStep(carrier, baseTimeStep, yearly=True)
            )
            for carrier,node in Element.createCustomSet(["setCarriers","setNodes"])
        )
    )

def constraintCarbonEmissionsCarrierRule(model, carrier, node, time):
    """ carbon emissions of importing/exporting carrier"""
    return (model.carbonEmissionsCarrier[carrier, node, time] ==
            model.carbonIntensityCarrier[carrier, node] *
            (model.importCarrierFlow[carrier, node, time] - model.exportCarrierFlow[carrier, node, time])
            )

def constraintCarbonEmissionsCarrierTotalRule(model, year):
    """ total carbon emissions of importing/exporting carrier"""
    baseTimeStep = EnergySystem.decodeTimeStep(None,year,"yearly")
    return(model.carbonEmissionsCarrierTotal[year] ==
        sum(
            sum(
                model.carbonEmissionsCarrier[carrier, node, time] * model.timeStepsOperationDuration[carrier, time]
                for time in EnergySystem.encodeTimeStep(carrier, baseTimeStep, yearly = True)
            )
            for carrier, node in Element.createCustomSet(["setCarriers", "setNodes"])
        )
    )

def constraintNodalEnergyBalanceRule(model, carrier, node, time):
    """" 
    nodal energy balance for each time step. 
    The constraint is indexed by setTimeStepsOperation, which is union of time step sequences of all corresponding technologies and carriers
    timeStepEnergyBalance --> baseTimeStep --> elementTimeStep
    """
    # decode to baseTimeStep
    baseTimeStep = EnergySystem.decodeTimeStep(carrier+"EnergyBalance",time)
    # carrier input and output conversion technologies
    carrierConversionIn, carrierConversionOut = 0, 0
    for tech in model.setConversionTechnologies:
        if carrier in model.setInputCarriers[tech]:
            elementTimeStep         = EnergySystem.encodeTimeStep(tech,baseTimeStep,"operation")
            carrierConversionIn     += model.inputFlow[tech,carrier,node,elementTimeStep]
        if carrier in model.setOutputCarriers[tech]:
            elementTimeStep         = EnergySystem.encodeTimeStep(tech,baseTimeStep,"operation")
            carrierConversionOut    += model.outputFlow[tech,carrier,node,elementTimeStep]
    # carrier flow transport technologies
    carrierFlowIn, carrierFlowOut   = 0, 0
    setEdgesIn                      = EnergySystem.calculateConnectedEdges(node,"in")
    setEdgesOut                     = EnergySystem.calculateConnectedEdges(node,"out")
    for tech in model.setTransportTechnologies:
        if carrier in model.setReferenceCarriers[tech]:
            elementTimeStep = EnergySystem.encodeTimeStep(tech,baseTimeStep,"operation")
            carrierFlowIn   += sum(model.carrierFlow[tech, edge, elementTimeStep]
                            - model.carrierLoss[tech, edge, elementTimeStep] for edge in setEdgesIn) 
            carrierFlowOut  += sum(model.carrierFlow[tech, edge, elementTimeStep] for edge in setEdgesOut) 
    # carrier flow storage technologies
    carrierFlowDischarge, carrierFlowCharge = 0, 0
    for tech in model.setStorageTechnologies:
        if carrier in model.setReferenceCarriers[tech]:
            elementTimeStep         = EnergySystem.encodeTimeStep(tech,baseTimeStep,"operation")
            carrierFlowDischarge    += model.carrierFlowDischarge[tech,node,elementTimeStep]
            carrierFlowCharge       += model.carrierFlowCharge[tech,node,elementTimeStep]
    # carrier import, demand and export
    carrierImport, carrierExport, carrierDemand = 0, 0, 0
    elementTimeStep     = EnergySystem.encodeTimeStep(carrier,baseTimeStep)
    carrierImport       = model.importCarrierFlow[carrier, node, elementTimeStep]
    carrierExport       = model.exportCarrierFlow[carrier, node, elementTimeStep]
    carrierDemand       = model.demandCarrier[carrier, node, elementTimeStep]
    
    return (
        # conversion technologies
        carrierConversionOut - carrierConversionIn 
        # transport technologies
        + carrierFlowIn - carrierFlowOut
        # storage technologies
        + carrierFlowDischarge - carrierFlowCharge
        # import and export 
        + carrierImport - carrierExport 
        # demand
        - carrierDemand 
        == 0
        )