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
import pyomo.environ            as pe
from ..energy_system        import EnergySystem
from .carrier               import Carrier
from ..component            import Parameter,Variable,Constraint

class ConditioningCarrier(Carrier):
    # set label
    label = "setConditioningCarriers"
    # empty list of elements
    listOfElements = []

    def __init__(self,carrier):
        """initialization of a generic carrier object
        :param carrier: carrier that is added to the model"""

        logging.info(f'Initialize conditioning carrier {carrier}')
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
        Variable.addVariable(
            model,
            name="endogenousCarrierDemand",
            indexSets= cls.createCustomSet(["setConditioningCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            doc = 'node- and time-dependent model endogenous carrier demand. \n\t Dimensions: setCarriers, setNodes, setTimeStepsCarrier. Domain: NonNegativeReals'
        )

    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <Carrier> """
        model = EnergySystem.getConcreteModel()

        # limit import flow by availability
        Constraint.addConstraint(
            model,
            name="constraintCarrierDemandCoupling",
            indexSets= cls.createCustomSet(["setConditioningCarrierParents","setNodes","setTimeStepsOperation"]),
            rule = constraintCarrierDemandCouplingRule,
            doc = 'coupeling model endogenous and exogenous carrier demand',
        )
        # overwrite energy balance when conditioning carriers are included
        model.constraintNodalEnergyBalance.deactivate()
        Constraint.addConstraint(
            model,
            name="constraintNodalEnergyBalanceConditioning",
            indexSets= cls.createCustomSet(["setCarriers", "setNodes", "setTimeStepsOperation"]),
            rule=constraintNodalEnergyBalanceWithConditioningRule,
            doc='node- and time-dependent energy balance for each carrier',
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
    params = Parameter.getComponentObject()

    # carrier input and output conversion technologies
    carrierConversionIn, carrierConversionOut = 0, 0
    for tech in model.setConversionTechnologies:
        if carrier in model.setInputCarriers[tech]:
            carrierConversionIn     += model.inputFlow[tech,carrier,node,time]
        if carrier in model.setOutputCarriers[tech]:
            carrierConversionOut    += model.outputFlow[tech,carrier,node,time]
    # carrier flow transport technologies
    carrierFlowIn, carrierFlowOut   = 0, 0
    setEdgesIn                      = EnergySystem.calculateConnectedEdges(node,"in")
    setEdgesOut                     = EnergySystem.calculateConnectedEdges(node,"out")
    for tech in model.setTransportTechnologies:
        if carrier in model.setReferenceCarriers[tech]:
            carrierFlowIn   += sum(model.carrierFlow[tech, edge, time]
                            - model.carrierLoss[tech, edge, time] for edge in setEdgesIn)
            carrierFlowOut  += sum(model.carrierFlow[tech, edge, time] for edge in setEdgesOut)
    # carrier flow storage technologies
    carrierFlowDischarge, carrierFlowCharge = 0, 0
    for tech in model.setStorageTechnologies:
        if carrier in model.setReferenceCarriers[tech]:
            carrierFlowDischarge    += model.carrierFlowDischarge[tech,node,time]
            carrierFlowCharge       += model.carrierFlowCharge[tech,node,time]
    # carrier import, demand and export
    carrierImport, carrierExport, carrierDemand = 0, 0, 0
    carrierImport           = model.importCarrierFlow[carrier, node, time]
    carrierExport           = model.exportCarrierFlow[carrier, node, time]
    carrierDemand           = params.demandCarrier[carrier, node, time]
    endogenousCarrierDemand = 0

    # check if carrier is conditioning carrier:
    if carrier in model.setConditioningCarriers:
        # check if carrier is parentCarrier of a conditioningCarrier
        if carrier in model.setConditioningCarrierParents:
            endogenousCarrierDemand = - model.endogenousCarrierDemand[carrier, node, time]
        else:
            endogenousCarrierDemand = model.endogenousCarrierDemand[carrier, node, time]

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