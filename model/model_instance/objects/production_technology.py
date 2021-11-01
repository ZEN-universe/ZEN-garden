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
        self.params = {
            'installProductionTech':    'installment of a production technology at node i and time t. Dimensions: setProductioTechnologies, setNodes, setTimeSteps'
        }
        if analysis['technologyApproximation'] == 'linear':
            self.params['converEfficiency']: 'Parameter which specifies the linear conversion efficiency of a technology.Dimensions: setTechnologies, setCarrierIn, setCarrierOut'
        elif analysis['technologyApproximation'] == 'PWA':
                 self.params['converEfficiency']: 'Parameter which specifies the linear conversion efficiency of a technology. Dimensions: setTechnologies, setCarrierIn, setCarrierOut, setSupportPointsPWA'

        # DECISION VARIABLES
        self.vars = {
            'importCarrier': 'node- and time-dependent carrier import from the grid. Dimensions: setCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals',
            'exportCarrier': 'node- and time-dependent carrier export from the grid. Dimensions: setCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals',
            'converEnergy': 'energy involved in conversion of carrier. Dimensions: setCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals'}
        # CONSTRAINTS
        self.constraints = {
            'constraint_technology_performance': '. Dimensions: setTechnologies, setCarriers, setAliasCarriers, setNodes, setTimeSteps'}

        self.addSet(self, sets)
        self.addParam(self.model, params)
        self.addVar(self.model, vars)
        self.addConstr(self.model, constraints)
        logging.info('added production technology sets, parameters, decision variables and constraints')


    # %% CONSTRAINTS
    def constraint_constraint_technology_performance(model, tech, carrierIn, carrierOut, node, time):
        """max carrier import from grid. Dimensions: setCarriers, setNodes, setTimeSteps"""

        return (model.converEfficiency[tech, carrierIn, carrierOut] * model.inputStream[tech, carrier, node, time]
                <= model.outputStream[tech, carrierOut, node, time])