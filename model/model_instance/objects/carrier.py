"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining a generic energy carrier.
              The class takes as inputs the abstract optimization model. The class adds parameters, variables and
              constraints of a generic carrier and returns the abstract optimization model.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
from model.model_instance.objects.element import Element

class Carrier(Element):

    def __init__(self, object):
        """initialization of a generic carrier object
        :param model: object of the abstract optimization model"""

        logging.info('initialize object of a generic carrier')
        super().__init__(object)

        # SETS AND SUBSETS
        subsets = {
            'setCarriersIn':        'Set of input carriers. Subset: setCarriers',
            'setCarriersOut':       'Set of output carriers. Subset: setCarriers',
            'setCarriersTransport': 'Set of carriers that can be transported. Subset: setCarriers'}
        self.addSubsets(subsets)

        # PARAMETERS
        params = {
            'demand':        'Parameter which specifies the carrier demand. Dimensions: setCarriers, setNodes, setTimeSteps',
            'price':         'Parameter which specifies the carrier price. Dimensions: setCarriers, setNodes, setTimeSteps',
            'cFootprint':    'Parameter which specifies the carbon intensity of a carrier. Dimensions: setCarriers',
            'cAvailability': 'Parameter which specifies the maximum energy that can be imported from the grid. Dimensions: setCarriersIn, setNodes, setTimeSteps'}
        self.addParams(params)

        # VARIABLES
        vars = {
            'importCarrier': 'node- and time-dependent carrier import from the grid. Dimensions: setCarriersIn, setNodes, setTimeSteps. Domain: NonNegativeReals'}
            #'exportCarrier': 'node- and time-dependent carrier export from the grid. Dimensions: setCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals'
            #todo add conversion energy / conditioning of carriers
            #'converEnergy':  'energy involved in conversion of carrier. Dimensions: setCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals'
        self.addVars(vars)

        # CONSTRAINTS
        constr = {
            'constraintAvailabilityCarrier': 'node- and time-dependent carrier availability. Dimensions: setCarriersIn, setNodes, setTimeSteps',
            'constraintNodalMassBalance':    'nodal mass balance for each time step. Dimensions: setCarriers, setNodes, setTimeSteps'}
        self.addConstr(constr)

        logging.info('added carrier sets, parameters, decision variables and constraints')

    #%% CONSTRAINTS
    def constraintAvailabilityCarrierRule(model, carrier, node, time):
        """node- and time-dependent carrier availability. Dimensions: setCarriers, setNodes, setTimeSteps"""

        return(model.importCarrier[carrier, node, time] <= model.cAvailability[carrier,node,time])

    def constraintNodalMassBalanceRule(model, carrier, node, time):
        """"nodal mass balance for each time step. Dimensions: setCarriers, setNodes, setTimeSteps"""

        carrierImport = 0
        if hassattr(model, 'setCarrier'):
            carrierImport = model.importCarrier[carrier, node, time], model.exportCarrier[carrier, node, time]

        carrierProductionIn, carrierProductionOut = 0
        if hassattr(model, 'setProduction'):
            if carrier in model.setCarrierIn:
                carrierProductionIn  = sum(model.outputProductionTech[tech, carrier, node, time] for tech in model.setProductionTech)
            if carrier in model.setCarrierOut:
                carrierProductionOut = sum(-model.outputProductionTech[tech, carrier, node, time] for tech in model.setProductionTech)

        carrierFlowIn, carrierFlowOut = 0
        if hassattr(model, 'setTransport'):
            carrierFlowIn  =  sum(sum(model.flowTransportTech[tech, carrier, aliasNode, node, time] for aliasNode in model.setAliasNodes) for tech in model.setTransportTech)
            carrierFlowOut =  sum(sum(model.flowTransportTech[tech, carrier, node, aliasNode, time] for aliasNode in model.setAliasNodes) for tech in model.setTransportTech)
        #TODO implement storage

        return (carrierImport - carrierExport
                + carrierProductionIn - carrierProductionOut
                + carrierFlowIn - carrierFlowOut
                - model.demand[carrier, node, time]
                == 0)