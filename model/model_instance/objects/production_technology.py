"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining the parameters, variables and constraints that hold for all technologies.
              The class takes as inputs the abstract optimization model. The class adds parameters, variables and
              constraints of a generic carrier and returns the abstract optimization model.
==========================================================================================================================================================================="""

import logging
import pyomo.environ as pe

class ProductionTechnology(Technology):

    def __init__(self, model):
        """init generic technology object"""

        logging.info('initialize object of a production technology')
        super.__init__(model)

        # SETS AND SUBSETS
        sets = {
            'setProductionTechnologies':  'Set of production technologies: Subset: setTechnologies'}
        if analysis['technologyApproximation'] == 'PWA':
            self.sets['setSupportPointsPWA'] = 'Set of support points for piecewise affine linearization'

        # PARAMETERS
        params = {
            'installProductionTech':    'installment of a production technology at node i and time t. Dimensions: setProductioTechnologies, setNodes, setTimeSteps'
        }
        if analysis['technologyApproximation'] == 'linear':
            params['converEfficiency']: 'Parameter which specifies the linear conversion efficiency of a technology.Dimensions: setTechnologies, setCarrierIn, setCarrierOut'
        elif analysis['technologyApproximation'] == 'PWA':
                 params['converEfficiency']: 'Parameter which specifies the linear conversion efficiency of a technology. Dimensions: setTechnologies, setCarrierIn, setCarrierOut, setSupportPointsPWA'
        techParams = self.getTechParams('TransportTech')
        self.params = {**techParams, **params}

        # DECISION VARIABLES
        self.vars = {}
        techVars = self.getTechVars('TransportTech')
        self.vars = {**techVars, **vars}
        #TODO implement conditioning for e.g. hydrogen
        #'converEnergy': 'energy involved in conversion of carrier. Dimensions: setCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals'

        # CONSTRAINTS
        self.constraints = {
            'constraintProductionTechPerformance': 'Conversion efficiency of production technology. Dimensions: setProductionTech, setCarriersIn, setCarriersOut, setNodes, setTimeSteps'}

        self.addSet(self, sets)
        self.addParam(self.model, params)
        self.addVar(self.model, vars)
        self.addConstr(self.model, constraints)
        logging.info('added production technology sets, parameters, decision variables and constraints')


    #%% CONSTRAINTS
    def constraintProductionTechPerformanceRule(model, tech, carrierIn, carrierOut, node, time):
        """conversion efficiency of production technology. Dimensions: setProductionTech, setCarriersIn, setNodes, setTimeSteps"""

        return (model.converEfficiency[tech, carrierIn, carrierOut] * model.inputStream[tech, carrierIn, node, time]
                <= model.outputStream[tech, carrierOut, node, time])


    # pre-defined in Technology class
    def constraintProductionTechSizeRule(model, tech, node, time):
        """min and max size of production technology. Dimensions: setProductionTech, setNodes, setTimeSteps"""

        return (model.maxSizeProductionTech[tech] * model.installProductionTech[tech, node, time], # lb
                model.sizeProductionTech[tech, node],                                              # expr
                model.maxSizeProductionTech[tech] * model.installProductionTech[tech, node, time]) # ub

    def constraintMinLoadProductionTech1Rule(model, carrier, tech, node, aliasNode, time):
        """min amount of carrier transported with transport technology between two nodes. Dimensions: setCarrier, setTransportTech, setNodes, setAlias, setTimeSteps"""
        return (model.flowLimit[transportTech, node, aliasNode, time] * model.minSizeTransportTech[transportTech], # lb
                model.carrierFlowAux[carrier, transportTech, node, aliasNode, time],                               # expr
                model.flowLimit[transportTech, node, aliasNode, time] * model.maxSizeTransportTech[transportTech]) # ub

    def constraintMinLoadProductionTech2Rule(model, carrier, tech, node, time):
        """min amount of carrier transported with transport technology between two nodes. Dimensions: setCarrier, setTransportTech, setNodes, setAlias, setTimeSteps"""
        return (model.carrierFlow - model.maxSizeTransportTech[tech] * (1 - model.flowLimit[tech, node, aliasNode, time]), # lb
                model.carrierFlowAux[carrier, tech, node, aliasNode, time],                                                # expr
                model.carrierFlow[carrier, tech, node, aliasNode, time])                                                   # ub

    def constraintMaxLoadProductionTechRule(model, carrier, tech, node, time):
        """min amount of carrier transported with transport technology between two nodes. Dimensions: setCarrier, setTransportTech, setNodes, setAlias, setTimeSteps"""
        return (model.outputProductionTech[carrier, tech, node, time] <= model.sizeProductionTech[tech, node, time])

    def constraintAvailabilityProductionTechRule(model, tech, node, time):
        """limited availability of production technology. Dimensions: setProductionTechnologies, setNodes, setTimeSteps"""
        return (model.availabilityProductionTech[tech, node, time] <= model.installProductionTech[tech, node, time])
