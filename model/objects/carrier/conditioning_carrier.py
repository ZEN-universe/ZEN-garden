"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class defining compressable energy carriers.
                The class takes as inputs the abstract optimization model. The class adds parameters, variables and
                constraints of a generic carrier and returns the abstract optimization model.
==========================================================================================================================================================================="""
import logging
import pyomo.environ                     as pe
from model.objects.energy_system         import EnergySystem
from model.objects.carrier.carrier       import Carrier

class ConditioningCarrier(Carrier):
    # set label
    label = "setConditioningCarriers"
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
        ConditioningCarrier.addElement(self)

    def storeInputData(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().storeInputData()

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to Carrier --- ###
    @classmethod
    def constructVars(cls):
        """ constructs the pe.Vars of the class <Carrier> """
        model = EnergySystem.getConcreteModel()
        
        # flow of imported carrier
        model.endogenousCarrierDemand = pe.Var(
            cls.createCustomSet(["setConditioningCarriers","setNodes","setTimeStepsCarrier"]),
            domain = pe.NonNegativeReals,
            doc = 'node- and time-dependent model endogenous carrier demand. \n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier. Domain: NonNegativeReals'
        )

    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <Carrier> """
        model = EnergySystem.getConcreteModel()

        # limit import flow by availability
        model.constraintCarrierDemandCoupling = pe.Constraint(
            cls.createCustomSet(["setConditioningCarrierParents","setNodes","setTimeStepsCarrier"]),
            rule = constraintCarrierDemandCouplingRule,
            doc = 'coupeling model endogenous and exogenous carrier demand. Dimensions: setConditioningCarriers, setNodes, setTimeStepsCarrier',
        )
        # overwrite energy balance when conditioning carriers are included
        model.constraintNodalEnergyBalance.deactivate()
        model.constraintNodalEnergyBalanceConditioning = pe.Constraint(
            cls.createCustomSet(["setCarriers", "setNodes", "setTimeStepsEnergyBalance"]),
            rule=constraintNodalEnergyBalanceWithConditioningRule,
            doc='node- and time-dependent energy balance for each carrier. Dimensions: setCarriers, setNodes, setTimeStepsEnergyBalance',
        )

def constraintCarrierDemandCouplingRule(model, parentCarrier, node, time):
    """ sum conditioning Carriers"""

    return(model.endogenousCarrierDemand[parentCarrier,node,time] ==
           sum(model.endogenousCarrierDemand[conditioningCarrier,node,time]
                 for conditioningCarrier in model.setConditioningCarrierChildren[parentCarrier]))

def constraintNodalEnergyBalanceWithConditioningRule(model, carrier, node, time):
    """" 
    nodal energy balance for each time step. 
    The constraint is indexed by setTimeStepsCarrier, which is union of time step sequences of all corresponding technologies and carriers
    timeStepEnergyBalance --> baseTimeStep --> elementTimeStep
    """
    # decode to baseTimeStep
    baseTimeStep            = EnergySystem.decodeTimeStep(carrier + "EnergyBalance", time)
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
    elementTimeStep         = EnergySystem.encodeTimeStep(carrier,baseTimeStep)
    carrierImport           = model.importCarrierFlow[carrier, node, elementTimeStep]
    carrierExport           = model.exportCarrierFlow[carrier, node, elementTimeStep]
    carrierDemand           = model.demandCarrier[carrier, node, elementTimeStep]
    endogenousCarrierDemand = 0

    # check if carrier is conditioning carrier:
    if carrier in model.setConditioningCarriers:
        # check if carrier is parentCarrier of a conditioningCarrier
        if carrier in model.setConditioningCarrierParents:
            endogenousCarrierDemand = - model.endogenousCarrierDemand[carrier, node, elementTimeStep]
        else:
            endogenousCarrierDemand = model.endogenousCarrierDemand[carrier, node, elementTimeStep]

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
        - endogenousCarrierDemand - carrierDemand
        == 0
        )