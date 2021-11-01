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

class Technology:

    def __init__(self, model):
        """init generic technology object"""

        logging.info('initialize object of a generic technology')
        super.__init__(model)

    def addTechnologySets(self):
        """add technology sets"""

        if analysis['technologyApproximation'] == 'PWA':
            self.sets['setSupportPointsPWA'] = 'Set of support points for piecewise affine linearization'

    def addTechnologyParams(self):
        """add technology params"""

        self.params = {
            'typeIn':  'Binary parameter which specifies the input carrier of the technology. Dimensions: setTechnologies, setCarriers',
            'typeOUT': 'Binary parameter which specifies the output carrier of the technology. Dimensions: setTechnologies, setCarriers',
            'sizeMin': 'Parameter which specifies the minimum technology size that can be installed. Dimensions: setTechnologies, setCarriers',
            'sizeMax': 'Parameter which specifies the maximum technology size that can be installed. Dimensions: setTechnologies, setCarriers'}

        if analysis['technologyApproximation'] == 'linear':
            self.params['converEfficiencyL']: 'Parameter which specifies the linear conversion efficiency of a technology.Dimensions: setTechnologies, setCarriers, setAliasCarriers'
        elif analysis['technologyApproximation'] == 'PWA':
            self.params['converEfficiency']: 'Parameter which specifies the linear conversion efficiency of a technology. Dimensions: setTechnologies, setCarriers, setAliasCarriers, setSupportPointsPWA'

        for param, paramProperties in params.items():
            self.addParam(self.model, param, paramProperties)

    def addTechnologyVariables(self):
        """add technology variables"""

        self.vars = {'carrierIn':   '',
                     'carrierOut':  ''}

        for var, varProperties in vars.items():
            self.addVar(self.model, var, varProperties)

    def addTechnologyConstraints(self):
        """add technology constraints"""

        self.constraints = {
            'constraint_technology_performance': '. Dimensions: setTechnologies, setCarriers, setAliasCarriers, setNodes, setTimeSteps'}

        for constr, constrProperties in self.constraints.items():
            self.addConstr(self.model, constr, constrProperties)

    # %% CONSTRAINTS
    def constraint_constraint_technology_performance(model, tech, carrier, aliasCarrier, node, time):
        """max carrier import from grid. Dimensions: setCarriers, setNodes, setTimeSteps"""

            return (model.converEfficiency[tech, carrier, aliasCarrier] * model.carrierIn[tech, carrier, node, time]
                    <= model.carrierIn[tech, aliasCarrier, node, time])