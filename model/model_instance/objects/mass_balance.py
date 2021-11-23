"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      November-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class containing the mass balance.
==========================================================================================================================================================================="""
from model.model_instance.objects.element import Element
import pyomo.environ as pe

class MassBalance(Element):

    def __init__(self, object):
        """
        initialization of a generic carrier object
        :param model: object of the abstract optimization model
        """

        super().__init__(object)
        constraint = {}#'constraintNodalMassBalance':    'nodal mass balance for each time step. Dimensions: setCarriers, setNodes, setTimeSteps'}
        self.addConstr(constraint)

    @staticmethod
    def constraintNodalMassBalanceRule(model, carrier, node, time):
        """"
        nodal mass balance for each time step.
        Dimensions: setCarriers, setNodes, setTimeSteps
        """
        carrierImport, carrierExport = 0, 0
        if carrier in model.setInputCarriers:
            if hassattr(model, 'setCarriers'):
                carrierImport = model.importCarrier[carrier, node, time]

        demand = 0
        if carrier in model.setOutputCarriers:
            demand = model.demandCarrier[carrier, node, time]
            if hassattr(model, 'setCarriers'):
                carrierExport = model.exportCarrier[carrier, node, time]

        carrierProductionIn, carrierProductionOut = 0, 0
        if hassattr(model, 'setProductionTechnologies'):
            if carrier in model.setInputCarriers:
                carrierProductionIn = sum(model.outputProductionTechnologies[tech, carrier, node, time] for tech in
                                          model.setProductionTechnologies)
            if carrier in model.setOutputCarriers:
                carrierProductionOut = sum(-model.outputProductionTechnologies[tech, carrier, node, time] for tech in
                                           model.setProductionTechnologies)

        carrierFlowIn, carrierFlowOut = 0, 0
        if hassattr(model, 'setTransportTechnologies'):
            carrierFlowIn = sum(
                sum(model.flowTransportTech[tech, carrier, aliasNode, node, time] for aliasNode in model.setAliasNodes) for
                tech in model.setTransportTech)
            carrierFlowOut = sum(
                sum(model.flowTransportTech[tech, carrier, node, aliasNode, time] for aliasNode in model.setAliasNodes) for
                tech in model.setTransportTech)
        # TODO implement storage

        return (carrierImport - carrierExport
                + carrierProductionIn - carrierProductionOut
                + carrierFlowIn - carrierFlowOut
                - demand
                == 0)