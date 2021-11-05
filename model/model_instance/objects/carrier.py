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

        # SETS AND SUBSETS
        self.sets = {
            'setCarriersIn':  'Set of input carriers. Subset: setCarriers',
            'setCarriersOut': 'Set of output carriers. Subset: setCarriers'}
        self.addSets(self, self.set)

        # PARAMETERS
        self.params = {
            'demand':        'Parameter which specifies the carrier demand. Dimensions: setCarriers, setNodes, setTimeSteps',
            'price':         'Parameter which specifies the carrier price. Dimensions: setCarriers, setNodes, setTimeSteps',
            'cFootprint':    'Parameter which specifies the carbon intensity of a carrier. Dimensions: setCarriers',
            'cAvailability': 'Parameter which specifies the maximum energy that can be imported from the grid. Dimensions: setCarriersIn, setNodes, setTimeSteps'}
        self.addParams(self.model, self.params)

        # VARIABLES
        self.vars = {
            'importCarrier': 'node- and time-dependent carrier import from the grid. Dimensions: setCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals',
            'exportCarrier': 'node- and time-dependent carrier export from the grid. Dimensions: setCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals'}
            #todo add conversion energy / conditioning of carriers
            #'converEnergy':  'energy involved in conversion of carrier. Dimensions: setCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals'
        self.addVar(self.model, self.vars)

        # CONSTRAINTS
        self.constraints = {
            'constraintAvailabilityCarrier': 'node- and time-dependent carrier availability. Dimensions: setCarriersIn, setNodes, setTimeSteps',
            'constraintMassBalance':         'nodal mass balance for each time step. Dimensions: setCarriers, setNodes, setTimeSteps'}

        self.addConstr(self.model, self.constraints)

        logging.info('added carrier sets, parameters, decision variables and constraints')

    #%% CONSTRAINTS
    def constraintAvailabilityCarrierRule(model, carrier, node, time):
        """node- and time-dependent carrier availability. Dimensions: setCarriers, setNodes, setTimeSteps"""

        return(model.importCarrier[carrier, node, time] <= model.cAvailability[carrier,node,time])


def constraintMassBalanceRule(model, carrier, node, time):
    """node- and time-dependent carrier availability. Dimensions: setCarriers, setNodes, setTimeSteps"""

    return (model.importCarrier[carrier, node, time] <= model.cAvailability[carrier, node, time])