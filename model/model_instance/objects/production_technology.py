"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining the parameters, variables and constraints of the production technologies.
              The class takes the abstract optimization model as an input, and adds parameters, variables and
              constraints of the production technologies.
==========================================================================================================================================================================="""

import logging
import pyomo.environ as pe
from model.model_instance.objects.technology import Technology

class ProductionTechnology(Technology):

    def __init__(self, object):
        """init generic technology object"""

        logging.info('initialize object of a production technology')
        super().__init__(object, 'Production')

        # SETS AND SUBSETS
        subsets = {}
        if self.analysis['technologyApproximation'] == 'PWA':
            self.subsets['setSupportPointsPWA'] = 'Set of support points for piecewise affine linearization'
        subsets = {**subsets, **self.getTechSubsets()}
        self.addSubsets(subsets)

        # PARAMETERS
        params = {}
        if self.analysis['technologyApproximation'] == 'linear':
            params['converEfficiency']: 'Parameter which specifies the linear conversion efficiency of a technology.Dimensions: setTechnologies, setCarrierIn, setCarrierOut'
        elif self.analysis['technologyApproximation'] == 'PWA':
                 params['converEfficiency']: 'Parameter which specifies the linear conversion efficiency of a technology. Dimensions: setTechnologies, setCarrierIn, setCarrierOut, setSupportPointsPWA'
        params = {**params, **self.getTechParams()}
        self.addParams(params)

        # DECISION VARIABLES
        vars = {}
        vars = {**vars, **self.getTechVars()}
        self.addVars(vars)
        #TODO implement conditioning for e.g. hydrogen
        #'converEnergy': 'energy involved in conversion of carrier. Dimensions: setCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals'

        # CONSTRAINTS
        constr = {
            'constraintProductionTechPerformance': 'Conversion efficiency of production technology. Dimensions: setProduction, setCarriersIn, setCarriersOut, setNodes, setTimeSteps'}
        constr = {**constr, **self.getTechConstr()}
        self.addConstr(constr)


        logging.info('added production technology sets, parameters, decision variables and constraints')

    #%% CONSTRAINTS
    def constraintProductionTechPerformanceRule(model, tech, carrierIn, carrierOut, node, time):
        """conversion efficiency of production technology. Dimensions: setProduction, setCarriersIn, setNodes, setTimeSteps"""

        if model.converEfficiency[tech, carrierIn, carrierOut]>0:
            return (model.converEfficiency[tech, carrierIn, carrierOut] * model.inputProductionTech[tech, carrierIn, node, time]
                    <= model.outputProductionTech[tech, carrierOut, node, time])
        else:
            return(model.inputProductionTech[tech, carrierIn, node, time] == 0)


    # pre-defined in Technology class
    def constraintProductionTechSizeRule(model, tech, node, time):
        """min and max size of production technology. Dimensions: setProduction, setNodes, setTimeSteps"""

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
        if model.converEfficiency[tech, carrierIn, carrierOut] > 0:
            return (model.outputProductionTech[carrier, tech, node, time] <= model.sizeProductionTech[tech, node, time])
        else:
            return (model.outputProductionTech[carrier, tech, node, time] == 0)

    def constraintAvailabilityProductionTechRule(model, tech, node, time):
        """limited availability of production technology. Dimensions: setProduction, setNodes, setTimeSteps"""
        return (model.availabilityProductionTech[tech, node, time] <= model.installProductionTech[tech, node, time])
