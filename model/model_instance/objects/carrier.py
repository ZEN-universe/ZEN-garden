"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining a generic energy carrier.
              The class takes as inputs the abstract optimization model. The class adds parameters, variables and
              constraints of a generic carrier and returns the abstract optimization model.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
from model.model_instance.objects.element import Element

class Carrier(Element):

    def __init__(self, object):
        """
        initialization of a generic carrier object
        :param model: object of the abstract optimization model
        """

        logging.info('initialize object of a generic carrier')
        super().__init__(object)

        # Subsets of carriers
        subsets = {
            'setInputCarriers': 'Set of input carriers. Subset: setCarriers',
            'setOutputCarriers': 'Set of output carriers. Subset: setCarriers',
            'setTransportCarriers': 'Set of carriers that can be transported. Subset: setCarriers'
            }
        self.addSubsets(subsets)

        # Parameters
        params = {
            'demandCarrier': 'Parameter which specifies the carrier demand. Dimensions: setOutputCarriers, setNodes, setTimeSteps',
            'exportPriceCarrier': 'Parameter which specifies the export carrier price. Dimensions: setCarriers, setNodes, setTimeSteps',
            'importPriceCarrier': 'Parameter which specifies the import carrier price. Dimensions: setCarriers, setNodes, setTimeSteps',            
            'footprintCarrier': 'Parameter which specifies the carbon intensity of a carrier. Dimensions: setCarriers',
            'availabilityCarrier': 'Parameter which specifies the maximum energy that can be imported from the grid. Dimensions: setInputCarriers, setNodes, setTimeSteps'
            }
        self.addParams(params)

        # Variables
        variables = {
            'importCarrier': 'node- and time-dependent carrier import from the grid. Dimensions: setInputCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals',
            'exportCarrier': 'node- and time-dependent carrier export from the grid. Dimensions: setOutputCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals'
            }
        self.addVars(variables)

        # Constraints
        constr = {
            'constraintAvailabilityCarrier': 'node- and time-dependent carrier availability. Dimensions: setInputCarriers, setNodes, setTimeSteps'
            }
        self.addConstr(constr)

        logging.info('added carrier sets, parameters, decision variables and constraints')
    
    @staticmethod
    def constraintAvailabilityCarrierRule(model, carrier, node, time):
        """
        node- and time-dependent carrier availability. 
        Dimensions: setCarriers, setNodes, setTimeSteps
        """
    
        return(model.importCarrier[carrier, node, time] <= model.availabilityCarrier[carrier,node,time])
    
    
    
    