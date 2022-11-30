"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class defining a generic energy carrier.
                The class takes as inputs the abstract optimization model. The class adds parameters, variables and
                constraints of a generic carrier and returns the abstract optimization model.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
import numpy as np
import pandas as pd
from ..element import Element
from ..energy_system import EnergySystem
from ..technology.technology import Technology
from ..component import Parameter,Variable,Constraint

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
        setTimeStepsYearly              = EnergySystem.getEnergySystem().setTimeStepsYearly
        # set attributes of carrier
        # raw import
        self.rawTimeSeries                              = {}
        self.rawTimeSeries["demandCarrier"]             = self.dataInput.extractInputData("demandCarrier",indexSets = ["setNodes","setTimeSteps"],timeSteps=setBaseTimeStepsYearly)
        self.rawTimeSeries["availabilityCarrierImport"] = self.dataInput.extractInputData("availabilityCarrier",indexSets = ["setNodes","setTimeSteps"],column="availabilityCarrierImport",timeSteps=setBaseTimeStepsYearly)
        self.rawTimeSeries["availabilityCarrierExport"] = self.dataInput.extractInputData("availabilityCarrier",indexSets = ["setNodes","setTimeSteps"],column="availabilityCarrierExport",timeSteps=setBaseTimeStepsYearly)
        self.rawTimeSeries["exportPriceCarrier"]        = self.dataInput.extractInputData("priceCarrier",indexSets = ["setNodes","setTimeSteps"],column="exportPriceCarrier",timeSteps=setBaseTimeStepsYearly)
        self.rawTimeSeries["importPriceCarrier"]        = self.dataInput.extractInputData("priceCarrier",indexSets = ["setNodes","setTimeSteps"],column="importPriceCarrier",timeSteps=setBaseTimeStepsYearly)
        # non-time series input data
        self.availabilityCarrierImportYearly            = self.dataInput.extractInputData("availabilityCarrierYearly",indexSets = ["setNodes","setTimeSteps"],column="availabilityCarrierImportYearly",timeSteps=setTimeStepsYearly)
        self.availabilityCarrierExportYearly            = self.dataInput.extractInputData("availabilityCarrierYearly",indexSets = ["setNodes","setTimeSteps"],column="availabilityCarrierExportYearly",timeSteps=setTimeStepsYearly)
        self.carbonIntensityCarrier                     = self.dataInput.extractInputData("carbonIntensity",indexSets = ["setNodes","setTimeSteps"],timeSteps=setTimeStepsYearly)
        self.shedDemandPrice                            = self.dataInput.extractInputData("shedDemandPrice",indexSets = [])
        
    def overwriteTimeSteps(self,baseTimeSteps):
        """ overwrites setTimeStepsOperation and  setTimeStepsEnergyBalance"""
        setTimeStepsOperation       = EnergySystem.encodeTimeStep(self.name, baseTimeSteps=baseTimeSteps, timeStepType="operation",yearly=True)
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
        # demand of carrier
        Parameter.addParameter(
            name="demandCarrier",
            data= EnergySystem.initializeComponent(cls,"demandCarrier",indexNames=["setCarriers","setNodes","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the carrier demand')
        # availability of carrier
        Parameter.addParameter(
            name="availabilityCarrierImport",
            data= EnergySystem.initializeComponent(cls,"availabilityCarrierImport",indexNames=["setCarriers","setNodes","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the maximum energy that can be imported from outside the system boundaries')
        # availability of carrier
        Parameter.addParameter(
            name="availabilityCarrierExport",
            data= EnergySystem.initializeComponent(cls,"availabilityCarrierExport",indexNames=["setCarriers","setNodes","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the maximum energy that can be exported to outside the system boundaries')
        # availability of carrier
        Parameter.addParameter(
            name="availabilityCarrierImportYearly",
            data= EnergySystem.initializeComponent(cls,"availabilityCarrierImportYearly",indexNames=["setCarriers","setNodes","setTimeStepsYearly"]),
            doc = 'Parameter which specifies the maximum energy that can be imported from outside the system boundaries for the entire year')
        # availability of carrier
        Parameter.addParameter(
            name="availabilityCarrierExportYearly",
            data= EnergySystem.initializeComponent(cls,"availabilityCarrierExportYearly",indexNames=["setCarriers","setNodes","setTimeStepsYearly"]),
            doc = 'Parameter which specifies the maximum energy that can be exported to outside the system boundaries for the entire year')
        # import price
        Parameter.addParameter(
            name="importPriceCarrier",
            data= EnergySystem.initializeComponent(cls,"importPriceCarrier",indexNames=["setCarriers","setNodes","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the import carrier price')
        # export price
        Parameter.addParameter(
            name="exportPriceCarrier",
            data= EnergySystem.initializeComponent(cls,"exportPriceCarrier",indexNames=["setCarriers","setNodes","setTimeStepsOperation"]),
            doc = 'Parameter which specifies the export carrier price')
        # demand shedding price
        Parameter.addParameter(
            name="shedDemandPrice",
            data=EnergySystem.initializeComponent(cls, "shedDemandPrice"),
            doc='Parameter which specifies the price to shed demand')
        # carbon intensity
        Parameter.addParameter(
            name="carbonIntensityCarrier",
            data= EnergySystem.initializeComponent(cls,"carbonIntensityCarrier",indexNames=["setCarriers","setNodes","setTimeStepsYearly"]),
            doc = 'Parameter which specifies the carbon intensity of carrier')

    @classmethod
    def constructVars(cls):
        """ constructs the pe.Vars of the class <Carrier> """
        model = EnergySystem.getConcreteModel()
        
        # flow of imported carrier
        model.importCarrierFlow = pe.Var(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            doc = 'node- and time-dependent carrier import from the grid'
        )
        # flow of exported carrier
        model.exportCarrierFlow = pe.Var(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            doc = 'node- and time-dependent carrier export from the grid'
        )
        # carrier import/export cost
        model.costCarrier = pe.Var(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.Reals,
            doc = 'node- and time-dependent carrier cost due to import and export'
        )
        # total carrier import/export cost
        model.costCarrierTotal = pe.Var(
            model.setTimeStepsYearly,
            domain = pe.Reals,
            doc = 'total carrier cost due to import and export'
        )
        # carbon emissions
        model.carbonEmissionsCarrier = pe.Var(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.Reals,
            doc = "carbon emissions of importing/exporting carrier"
        )
        # carbon emissions carrier
        model.carbonEmissionsCarrierTotal = pe.Var(
            model.setTimeStepsYearly,
            domain=pe.Reals,
            doc="total carbon emissions of importing/exporting carrier"
        )
        # shed demand
        model.shedDemandCarrier = pe.Var(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            domain=pe.NonNegativeReals,
            doc="shed demand of carrier"
        )
        # cost of shed demand
        model.costShedDemandCarrier = pe.Var(
            cls.createCustomSet(["setCarriers", "setNodes", "setTimeStepsOperation"]),
            domain=pe.NonNegativeReals,
            doc="shed demand of carrier"
        )

        # add pe.Sets of the child classes
        for subclass in cls.getAllSubclasses():
            if np.size(EnergySystem.getSystem()[subclass.label]):
                subclass.constructVars()

    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <Carrier> """
        model = EnergySystem.getConcreteModel()

        # limit import flow by availability
        model.constraintAvailabilityCarrierImport = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            rule = constraintAvailabilityCarrierImportRule,
            doc = 'node- and time-dependent carrier availability to import from outside the system boundaries',
        )        
        # limit export flow by availability
        model.constraintAvailabilityCarrierExport = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            rule = constraintAvailabilityCarrierExportRule,
            doc = 'node- and time-dependent carrier availability to export to outside the system boundaries',
        )
        # limit import flow by availability for each year
        model.constraintAvailabilityCarrierImportYearly = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsYearly"]),
            rule = constraintAvailabilityCarrierImportYearlyRule,
            doc = 'node- and time-dependent carrier availability to import from outside the system boundaries summed over entire year',
        )
        # limit export flow by availability for each year
        model.constraintAvailabilityCarrierExportYearly = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsYearly"]),
            rule = constraintAvailabilityCarrierExportYearlyRule,
            doc = 'node- and time-dependent carrier availability to export to outside the system boundaries, summed over entire year',
        )
        # cost for carrier
        model.constraintCostCarrier = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            rule = constraintCostCarrierRule,
            doc = "cost of importing/exporting carrier"
        )
        # cost for carrier
        model.constraintCostShedDemand = pe.Constraint(
            cls.createCustomSet(["setCarriers", "setNodes", "setTimeStepsOperation"]),
            rule=constraintCostShedDemandRule,
            doc="cost of shedding carrier demand"
        )
        # total cost for carriers
        model.constraintCostCarrierTotal = pe.Constraint(
            model.setTimeStepsYearly,
            rule = constraintCostCarrierTotalRule,
            doc = "total cost of importing/exporting carriers"
        )
        # carbon emissions
        model.constraintCarbonEmissionsCarrier = pe.Constraint(
            cls.createCustomSet(["setCarriers","setNodes","setTimeStepsOperation"]),
            rule = constraintCarbonEmissionsCarrierRule,
            doc = "carbon emissions of importing/exporting carrier"
        )

        # carbon emissions carrier
        model.constraintCarbonEmissionsCarrierTotal = pe.Constraint(
            model.setTimeStepsYearly,
            rule=constraintCarbonEmissionsCarrierTotalRule,
            doc="total carbon emissions of importing/exporting carriers"
        )
        # energy balance
        model.constraintNodalEnergyBalance = pe.Constraint(
            cls.createCustomSet(["setCarriers", "setNodes", "setTimeStepsEnergyBalance"]),
            rule=constraintNodalEnergyBalanceRule,
            doc='node- and time-dependent energy balance for each carrier',
        )
        # add pe.Sets of the child classes
        for subclass in cls.getAllSubclasses():
            if np.size(EnergySystem.getSystem()[subclass.label]):
                subclass.constructConstraints()


#%% Constraint rules defined in current class
def constraintAvailabilityCarrierImportRule(model, carrier, node, time):
    """node- and time-dependent carrier availability to import from outside the system boundaries"""
    # get parameter object
    params = Parameter.getComponentObject()
    if params.availabilityCarrierImport[carrier,node,time] != np.inf:
        return(model.importCarrierFlow[carrier, node, time] <= params.availabilityCarrierImport[carrier,node,time])
    else:
        return pe.Constraint.Skip

def constraintAvailabilityCarrierExportRule(model, carrier, node, time):
    """node- and time-dependent carrier availability to export to outside the system boundaries"""
    # get parameter object
    params = Parameter.getComponentObject()
    if params.availabilityCarrierExport[carrier,node,time] != np.inf:
        return(model.exportCarrierFlow[carrier, node, time] <= params.availabilityCarrierExport[carrier,node,time])
    else:
        return pe.Constraint.Skip

def constraintAvailabilityCarrierImportYearlyRule(model, carrier, node, year):
    """node- and year-dependent carrier availability to import from outside the system boundaries"""
    # get parameter object
    params = Parameter.getComponentObject()
    baseTimeStep = EnergySystem.decodeTimeStep(None, year, "yearly")
    if params.availabilityCarrierImportYearly[carrier,node,year] != np.inf:
        return(
            params.availabilityCarrierImportYearly[carrier, node, year] >=
            sum(
                model.importCarrierFlow[carrier, node, time]
                * params.timeStepsOperationDuration[carrier, time]
                for time in EnergySystem.encodeTimeStep(carrier, baseTimeStep, yearly=True)
                )
        )
    else:
        return pe.Constraint.Skip

def constraintAvailabilityCarrierExportYearlyRule(model, carrier, node, year):
    """node- and year-dependent carrier availability to export to outside the system boundaries"""
    # get parameter object
    params = Parameter.getComponentObject()
    baseTimeStep = EnergySystem.decodeTimeStep(None, year, "yearly")
    if params.availabilityCarrierExportYearly[carrier,node,year] != np.inf:
        return (
            params.availabilityCarrierExportYearly[carrier, node, year] >=
            sum(
                model.exportCarrierFlow[carrier, node, time]
                * params.timeStepsOperationDuration[carrier, time]
                for time in EnergySystem.encodeTimeStep(carrier, baseTimeStep, yearly=True)
            )
        )
    else:
        return pe.Constraint.Skip

def constraintCostCarrierRule(model, carrier, node, time):
    """ cost of importing/exporting carrier"""
    # get parameter object
    params = Parameter.getComponentObject()
    return(model.costCarrier[carrier,node, time] ==
        params.importPriceCarrier[carrier, node, time]*model.importCarrierFlow[carrier, node, time] -
        params.exportPriceCarrier[carrier, node, time]*model.exportCarrierFlow[carrier, node, time]
    )

def constraintCostShedDemandRule(model, carrier, node, time):
    """ cost of shedding demand of carrier """
    # get parameter object
    params = Parameter.getComponentObject()
    if params.shedDemandPrice[carrier] != np.inf:
        return(
            model.costShedDemandCarrier[carrier,node, time] ==
            model.shedDemandCarrier[carrier,node,time] * params.shedDemandPrice[carrier]
        )
    else:
        return(
            model.shedDemandCarrier[carrier, node, time] == 0
        )

def constraintCostCarrierTotalRule(model,year):
    """ total cost of importing/exporting carrier"""
    # get parameter object
    params = Parameter.getComponentObject()
    baseTimeStep = EnergySystem.decodeTimeStep(None, year, "yearly")
    return(model.costCarrierTotal[year] ==
        sum(
            sum(
                (model.costCarrier[carrier,node,time] + model.costShedDemandCarrier[carrier,node, time])
                * params.timeStepsOperationDuration[carrier, time]
                for time in EnergySystem.encodeTimeStep(carrier, baseTimeStep, yearly=True)
            )
            for carrier,node in Element.createCustomSet(["setCarriers","setNodes"])
        )
    )

def constraintCarbonEmissionsCarrierRule(model, carrier, node, time):
    """ carbon emissions of importing/exporting carrier"""
    # get parameter object
    params = Parameter.getComponentObject()
    baseTimeStep    = EnergySystem.decodeTimeStep(carrier, time)
    yearlyTimeStep  = EnergySystem.encodeTimeStep(None,baseTimeStep,"yearly")
    return (model.carbonEmissionsCarrier[carrier, node, time] ==
            params.carbonIntensityCarrier[carrier, node, yearlyTimeStep] *
            (model.importCarrierFlow[carrier, node, time] - model.exportCarrierFlow[carrier, node, time])
            )

def constraintCarbonEmissionsCarrierTotalRule(model, year):
    """ total carbon emissions of importing/exporting carrier"""
    # get parameter object
    params = Parameter.getComponentObject()
    baseTimeStep = EnergySystem.decodeTimeStep(None,year,"yearly")
    return(model.carbonEmissionsCarrierTotal[year] ==
        sum(
            sum(
                model.carbonEmissionsCarrier[carrier, node, time] * params.timeStepsOperationDuration[carrier, time]
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
    # get parameter object
    params = Parameter.getComponentObject()
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
    carrierDemand       = params.demandCarrier[carrier, node, elementTimeStep]
    # shed demand
    carrierShedDemand   = model.shedDemandCarrier[carrier, node, elementTimeStep]
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
        # shed demand
        + carrierShedDemand
        == 0
    )
