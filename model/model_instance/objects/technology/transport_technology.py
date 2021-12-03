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
from model.model_instance.objects.technology.technology import Technology

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
                                     Dimensions: setTransportTechnologies'}
        params = {**params, **self.getTechParams()}
        self.addParams(params)

        # VARIABLES
        vars = {
            'carrierFlow':            'carrier flow through transport technology from node i to node j. \
                                       Dimensions: setTransportCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps.\
                                       Domain: NonNegativeReals',
            'carrierFlowAux':         'auxiliary variable to model the min possible flow through transport technology from node i to node j. \
                                       Dimensions: setTransportCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps.\
                                       Domain: NonNegativeReals'}
        vars = {**vars, **self.getTechVars()}
        self.addVars(vars)


        # CONSTRAINTS
        constr = {'constraintTransportTechnologiesPerformance':  'performance of transport technology. \
                                                                  Dimensions: setTransportCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps',
                  'constraintMinLoadTransportTechnologies1':     'min flow through transport technology, part one. \
                                                                  Dimensions: setTransportCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps',
                  'constraintMinLoadTransportTechnologies2':     'min flow through transport, part two. \
                                                                  Dimensions: setTransportCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps',
                  'constraintMaxLoadTransportTechnologies1':     'max flow through transport technology, part one. \
                                                                  Dimensions: setTransportCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps',
                  'constraintMaxLoadTransportTechnologies2':     'max flow through transport, part two. \
                                                                  Dimensions: setTransportCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps'

                  }
        # TODO add constraints for transport losses
        constr = {**constr, **self.getTechConstr()}
        #self.addConstr(constr)

        logging.info('added transport technology sets, parameters, decision variables and constraints')

    @staticmethod
    def constraintTransportTechnologiesPerformanceRule(model, carrier, tech, node, aliasNode, time):
        """constraint to account for transport losses. Dimensions: setTransportTechnologies, setTransportCarriers, setNodes, setTimeSteps"""
        # TODO implement transport losses
        return (model.carrierFlow[carrier, tech, node, aliasNode, time]
                == model.carrierFlow[carrier,tech, aliasNode, node, time])
    
    # pre-defined in Technology class
    # capacity constraints
    @staticmethod
    def constraintTransportTechnologiesMinCapacityRule(model, tech, node, aliasNode, time):
        """min size of transport technology. Dimensions: setTransportTechnologiesnologies, setNodes, setAliasNodes, setTimeSteps"""

        return (model.minCapacityTransport[tech] * model.installTransportTechnologies[tech, node, time]
                <= model.capacityTransportTechnologies[tech, node, time])

    @staticmethod
    def constraintTransportTechnologiesMaxCapacityRule(model, tech, node, aliasNode, time):
        """max size of transport technology. Dimensions: setTransportTechnologiesnologies, setNodes, setAliasNodes, setTimeSteps"""
        return (model.maxCapacityTransport[tech] * model.installTransportTechnologies[tech, node, time]
                >= model.capacityTransportTechnologies[tech, node, time])

    # operational constraints
    @staticmethod
    def constraintMinLoadTransportTechnologies1Rule(model, carrier, tech, node, aliasNode, time):
        """min flow through transport technology between two nodes. setTransportCarriers, setTransportTechnologiesnologies, setNodes, setAliasNodes, setTimeSteps"""
        return (model.minTransportTechLoad[tech] * model.installTransportTechnologies[tech, node, aliasNode, time] * model.minCapacityTransport[tech]
                <= model.carrierFlowAux[carrier, tech, node, aliasNode, time])

    def constraintMinLoadTransportTechnologies2Rule(model, carrier, tech, node, aliasNode, time):
        """min amount of carrier flow thorugh transport technology between two nodes. Dimensions: setCarrier, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps"""
        return (model.carrierFlow[carrier, tech, node, aliasNode, time] - model.maxCapacityTransport[tech] * (1 - model.installTransportTechnologies[tech, node, aliasNode, time])
                <= model.carrierFlowAux[carrier, tech, node, aliasNode, time])
    
    def constraintMaxLoadTransportTechnologies1Rule(model, carrier, tech, node, aliasNode, time):
        """max amount of carrier flow through transport technology between two nodes. Dimensions: setCarrier, setTransportTechnologiesnologies, setNodes, setAlias, setTimeSteps"""
        return (model.capacityTransportTechnologies[tech, node, aliasNode, time]
                >= model.carrierFlowAux[carrier, tech, node, aliasNode, time])                                           # ub
    
    def constraintMaxLoadTransportTechnologies2Rule(model, carrier, tech, node, aliasNode, time):
        """max amount of carrier flow through transport technology between two nodes. Dimensions: setCarrier, setTransportTechnologiesnologies, setNodes, setAlias, setTimeSteps"""
        return (model.carrierFlow[carrier, tech, node, time]
                >= model.carrierFlowAux[carrier, tech, node, time])

    def constraintAvailabilityTransportTechnologiesRule(model, tech, node, aliasNode, time):
        """limited availability of production technology. Dimensions: setProductionTechnologies, setNodes, setTimeSteps"""
        return (model.availabilityTransport[tech, node, aliasNode, time] <= model.installTransportTechnologies[tech, node, aliasNode, time])
