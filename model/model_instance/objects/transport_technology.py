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
        sets = {
            'setAliasNodes':     'Copy of the set of nodes to model transport. Subset: setNodes',
            'setTransportTech': 'Set of production technologies: Subset: setTechnologies'}

        # PARAMETERS
        self.params = {
            'installTransportTech': 'installment of a production technology at node i and time t. Dimensions: setTransportTech, setNodes, setTimeSteps'
        }

        # VARIABLES
        self.vars = {
            'flowCarrier': 'carrier flow through transport technology from node i to node j. Dimensions: setCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps'}

        logging.info('added transport technology')