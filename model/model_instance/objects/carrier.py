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
        """initialization of a generic carrier object
        :param model: object of the abstract optimization model"""

        logging.info('initialize object of a generic carrier')
        super().__init__(object)

        #%% Sets and subsets
        subsets = {
            'setInputCarriers':      'Set of technology specific input carriers. \
                                      \n\t Dimension: setCarriers',
            'setOutputCarriers':     'Set of technology specific output carriers. \
                                      \n\t Subset: setCarriers'}
        self.addSets(subsets)

        #%% Parameters
        params = {
            'demandCarrier':                    'Parameter which specifies the carrier demand. \
                                                \n\t Dimensions: setOutputCarriers, setNodes, setTimeSteps',
            'availabilityCarrier':              'Parameter which specifies the maximum energy that can be imported from the grid.\
                                                \n\t Dimensions: setInputCarriers, setNodes, setTimeSteps'
            # 'exportPriceCarrier': 'Parameter which specifies the export carrier price. \n\t Dimensions: setCarriers, setNodes, setTimeSteps',
            # 'importPriceCarrier': 'Parameter which specifies the import carrier price. \n\t Dimensions: setCarriers, setNodes, setTimeSteps',               
            # 'footprintCarrier': 'Parameter which specifies the carbon intensity of a carrier. \n\t Dimensions: setCarriers',s
            }
        self.addParams(params)

        #%% Variables
        variables = {
            'importCarrier':                    'node- and time-dependent carrier import from the grid.\
                                                \n\t Dimensions: setInputCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals',
            'exportCarrier':                    'node- and time-dependent carrier export from the grid. \
                                                \n\t Dimensions: setOutputCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals'
            }
        self.addVars(variables)

        #%% Contraints in current class
        constr = {
            'AvailabilityCarrier':    'node- and time-dependent carrier availability.\
                                       \n\t Dimensions: setInputCarriers, setNodes, setTimeSteps',
            }
        self.addConstr(constr)

        logging.info('added carrier sets, parameters, decision variables and constraints')

    #%% Rules contraints defined in current class
    @staticmethod
    def constraintAvailabilityCarrierRule(model, carrier, node, time):
        """node- and time-dependent carrier availability"""

        return(model.importCarrier[carrier, node, time] <= model.availabilityCarrier[carrier,node,time])
    
    
    
    
    
    