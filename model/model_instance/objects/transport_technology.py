"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining the parameters, variables and constraints that hold for all transport technologies.
              The class takes the abstract optimization model as an input, and returns the parameters, variables and
              constraints that hold for the transport technologies.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
from model.model_instance.objects.technology import Technology

class TransportTechnology(Technology):

    def __init__(self, object):
        """init generic technology object"""

        logging.info('initialize object of a transport technology')
        super().__init__(object, 'Transport')

        # SETS AND SUBSETS
        subsets = {
            'setAliasNodes': 'Copy of the set of nodes to model transport. Subset: setNodes'}
        subsets = {**subsets, **self.getTechSubsets()}
        self.addSubsets(subsets)


        # PARAMETERS
        params = {
            'minTransportTechLoad': 'fraction of installed transport technology size that determines the minimum load of the transport technology. \
                                     Dimensions: setTransport'}
        params = {**params, **self.getTechParams()}
        self.addParams(params)

        # VARIABLES
        vars = {
            'flowTransportTech':      'carrier flow through transport technology from node i to node j. \
                                       Dimensions: setCarriers, setTransport, setNodes, setAliasNodes, setTimeSteps.\
                                       Domain: NonNegativeReals',
            'carrierFlowAux':         'auxiliary variable to model the min possible flow through transport technology from node i to node j. \
                                       Dimensions: setCarriers, setTransport, setNodes, setAliasNodes, setTimeSteps.\
                                       Domain: NonNegativeReals',
            'flowLimitTransportTech': 'auxiliary variable to model the minimum flow through a transport technology between nodes. \
                                       Dimensions: setTransport, setNodes, setAliasNodes, setTimeSteps.\
                                       Domain: NonNegativeReals'}
        vars = {**vars, **self.getTechVars()}
        self.addVars(vars)


        # CONSTRAINTS
        constr = {'constraintTransportTechPerformance':  'performance of transport technology. Dimensions: setTransport, setCarriersTransport, setNodes, setTimeSteps'}
        # TODO add constraints for transport losses
        constr = {**constr, **self.getTechConstr()}
        self.addConstr(constr)

        logging.info('added transport technology sets, parameters, decision variables and constraints')

    #%% CONSTRAINTS
    def constraintTransportTechPerformanceRule(model, tech, carrier, node, time):
        """conversion efficiency of production technology. Dimensions: setProduction, setCarriersIn, setNodes, setTimeSteps"""
        # TODO implement transport losses
        return (model.carrierFlow[tech, carrier, node, aliasNode, time]
                == model.carrierFlow[tech, carrier, node, aliasNode, time])

    # pre-defined in Technology class
    def constraintTransportTechSizeRule(model, tech, node, aliasNode, time):
        """min size of transport technology that can be installed between two nodes. setTransport, setNodes, setAliasNodes, setTimeSteps"""
        return(model.installTransportTech[tech, node, aliasNode, time]*model.minSizeTransportTech[tech], # lb
               model.sizeTransportTech[tech, node, aliasNode, time],                                              # expr
               model.installTransportTech[tech, node, aliasNode, time]*model.maxSizeTransportTech[tech]) # ub

    def constraintMinLoadTransportTech1Rule(model, carrier, tech, node, aliasNode, time):
        """min amount of carrier transported with transport technology between two nodes. Dimensions: setCarrier, setTransport, setNodes, setAlias, setTimeSteps"""
        return(model.flowLimitTransportTech[tech,node, aliasNode, time] * model.minSizeTransportTech[tech], # lb
               model.carrierFlowAux[carrier, tech, node, aliasNode, time],                              # expr
               model.flowLimitTransportTech[tech,node, aliasNode, time] * model.maxSizeTransportTech[tech]) # ub

    def constraintMinLoadTransportTech2Rule(model, carrier, tech, node, aliasNode, time):
        """min amount of carrier transported with transport technology between two nodes. Dimensions: setCarrier, setTransport, setNodes, setAlias, setTimeSteps"""
        return(model.flowTransportTech[carrier, tech, node, aliasNode, time] - model.maxSizeTransportTech[tech]*(1-model.flowLimit[tech, node, aliasNode, time]),  # lb
               model.carrierFlowAux[carrier, tech, node, aliasNode, time],                                                     # expr
               model.carrierFlow[carrier, tech, node, aliasNode, time])                                                        # ub

    def constraintMaxLoadTransportTechRule(model, carrier, tech, node, aliasNode, time):
        """min amount of carrier transported with transport technology between two nodes. Dimensions: setCarrier, setTransport, setNodes, setAlias, setTimeSteps"""
        return(model.flowTransportTech[carrier, tech, node, aliasNode, time] <= model.sizeTransportTech[tech, node, aliasNode, time])

    def constraintAvailabilityTransportTechRule(model, tech, node, aliasNode, time):
        """limited availability of production technology. Dimensions: setProductionTechnologies, setNodes, setTimeSteps"""
        return (model.availabilityTransportTech[tech, node, aliasNode, time] <= model.installTransportTech[tech, node, aliasNode, time])
