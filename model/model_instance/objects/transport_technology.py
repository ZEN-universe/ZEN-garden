"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining the parameters, variables and constraints that hold for transport technologies.
              The class takes as inputs the abstract optimization model. The class adds parameters, variables and
              constraints of the transport technologies and returns the abstract optimization model.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe

class TransportTechnology(Technology):

    def __init__(self, model):
        """init generic technology object"""

        logging.info('initialize object of a transport technology')
        super.__init__(model)

        # SETS AND SUBSETS
        self.sets = {
            'setAliasNodes':     'Copy of the set of nodes to model transport. Subset: setNodes',
            'setTransportTech': 'Set of production technologies: Subset: setTechnologies'}

        # PARAMETERS
        params = {
            'minTransportTechLoad':     'fraction of installed transport technology size that determines the minimum load of the transport technology. Dimensions: setTransportTech',
            }
        techParams  = self.getTechParams('TransportTech')
        self.params = {**techParams, **params}

        # VARIABLES
        vars = {
            'flowTransportTech':      'carrier flow through transport technology from node i to node j. Dimensions: setCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps',
            'carrierFlowAux':         'auxilary variable to model the min possible flow through transport technology from node i to node j. Dimensions: setCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps',
            'flowLimitTransportTech': 'auxilary variable to model the minimum flow through a transport technology between nodes. Dimensions: setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps'}
        techVars = self.getTechVars('TransportTech')
        self.vars = {**techVars, **vars}

        # CONSTRAINTS
        constr = {}
        # TODO add constraints for transport losses
        techConstr = self.getTechConstraints('TransportTech')
        self.constraints = {**techConstr, **constr}

        logging.info('added transport technology sets, parameters, decision variables and constraints')

    #%% CONSTRAINTS
    # pre-defined in Technology class
    def constraintTransportTechSizeRule(model, tech, node, aliasNode, time):
        """min size of transport technology that can be installed between two nodes. setTransportTech, setNodes, setAliasNodes, setTimeSteps"""
        return(model.installTransportTech[tech, node, aliasNode, time]*model.minSizeTransportTech[tech], # lb
               model.sizeTransportTech[tech, node, aliasNode, time],                                              # expr
               model.installTransportTech[tech, node, aliasNode, time]*model.maxSizeTransportTech[tech]) # ub

    def constraintMinCarrierTransport1Rule(model, carrier, tech, node, aliasNode, time):
        """min amount of carrier transported with transport technology between two nodes. Dimensions: setCarrier, setTransportTech, setNodes, setAlias, setTimeSteps"""
        return(model.flowLimitTransportTech[tech,node, aliasNode, time] * model.minSizeTransportTech[tech], # lb
               model.carrierFlowAux[carrier, tech, node, aliasNode, time],                              # expr
               model.flowLimitTransportTech[tech,node, aliasNode, time] * model.maxSizeTransportTech[tech]) # ub

    def constraintMinCarrierTransport2Rule(model, carrier, tech, node, aliasNode, time):
        """min amount of carrier transported with transport technology between two nodes. Dimensions: setCarrier, setTransportTech, setNodes, setAlias, setTimeSteps"""
        return(model.flowTransportTech[carrier, tech, node, aliasNode, time] - model.maxSizeTransportTech[tech]*(1-model.flowLimit[tech, node, aliasNode, time]),  # lb
               model.carrierFlowAux[carrier, tech, node, aliasNode, time],                                                     # expr
               model.carrierFlow[carrier, tech, node, aliasNode, time])                                                        # ub

    def constraintMaxCarrierTransportRule(model, carrier, tech, node, aliasNode, time):
        """min amount of carrier transported with transport technology between two nodes. Dimensions: setCarrier, setTransportTech, setNodes, setAlias, setTimeSteps"""
        return(model.flowTransportTech[carrier, tech, node, aliasNode, time] <= model.sizeTransportTech[tech, node, aliasNode, time])

