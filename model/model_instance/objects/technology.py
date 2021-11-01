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

class Technology(Element):

    def __init__(self, model):
        """init generic technology object"""

        logging.info('initialize object of a generic technology')
        super.__init__(model)

        # PARAMETERS
        params = {
            'sizeMin': 'Parameter which specifies the minimum technology size that can be installed.',
            'sizeMax': 'Parameter which specifies the maximum technology size that can be installed.'}
        # VARIABLES
        vars = {
            'inputStream': 'Input stream of a carrier into a technology.',
            'outputStream': 'Output stream of a carrier into a technology.'}