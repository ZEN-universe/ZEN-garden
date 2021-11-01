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
from element import Element

class Carrier(Element):

    def __init__(self, model):
        """initialization of a generic carrier object
        :param model: object of the abstract optimization model"""

        logging.info('initialize object of a generic carrier')
        super.__init__(model)


    #%% METHODS
    def addCarrierSets(self):
        """add carrier subsets"""
        logging.info('add parameters of a generic carrier')

        self.sets = {
            'setAliasCarrier':  'Copy of the set of all carriers'}
            #'setGridIn':        'Set of all carriers with limited grid supply. Subset: setCarrier'}

        for set, setProperties in self.Sets.items():
            self.addSet(self, set, setProperties)

    def addCarrierParams(self):
        """add carrier parameters"""
        logging.info('add parameters of a generic carrier')

        self.params = {
            'demand': 'Parameter which specifies the carrier demand. Dimensions: setCarriers, setNodes, setTimeSteps',
            'price': 'Parameter which specifies the carrier price. Dimensions: setCarriers, setNodes, setTimeSteps',
            'cFootprint': 'Parameter which specifies the carbon intensity of a carrier. Dimensions: setCarriers',
            'gridIn': 'Parameter which specifies the maximum energy that can be imported from the grid. Dimensions: setCarriers, setNodes, setTimeSteps'}

        for param, paramProperties in self.params.items():
            self.addParam(self.model, param, paramProperties)


    def addCarrierVariables(self):
        """add carrier variables"""
        logging.info('add variables of a generic carrier')

        self.vars = {
            'importCarrier': 'node- and time-dependent carrier import from the grid. Dimensions: setCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals',
            'exportCarrier': 'node- and time-dependent carrier export from the grid. Dimensions: setCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals',
            'converEnergy': 'energy involved in conversion of carrier. Dimensions: setCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals'}


        for var, varProperties in self.vars.items():
            self.addVar(self.model, var, varProperties)

    def addCarrierConstraints(self):
        """add carrier constraints"""
        logging.info('add generic carrier constraints')

        self.constraints = {
            'constraint_max_carrier_import': 'max carrier import from grid. Dimensions: setNodes, setTimeSteps'}

        for constr, constrProperties in self.constraints.items():
            self.addConstr(self.model, constr, constrProperties)

    #%% CONSTRAINTS
    def constraint_max_carrier_import_rule(model, carrier, node, time):
        """max carrier import from grid. Dimensions: setCarriers, setNodes, setTimeSteps"""

        return(model.importCarrier[carrier, node, time] <= model.gridIn[carrier,node,time])




