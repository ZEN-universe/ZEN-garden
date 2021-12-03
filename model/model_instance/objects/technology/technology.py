"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining the parameters, variables and constraints that hold for all technologies.
              The class takes the abstract optimization model as an input, and returns the parameters, variables and
              constraints that hold for all technologies.
==========================================================================================================================================================================="""

import logging
import pyomo.environ as pe
from model.model_instance.objects.element import Element

class Technology(Element):

    def __init__(self, object, technologyType):
        """init generic technology object"""

        logging.info('initialize object of a generic technology')
        super().__init__(object)
        self.type = technologyType
        self.dim  = self.getDimensions()

    def getDimensions(self):

        if self.type == 'Transport':
            dim = 'setNodes, setAliasNodes'
        else:
            dim = 'setNodes'

        return dim

    def getTechSubsets(self):

        subsets = {
            f'set{self.type}Technologies': f'Set of {self.type} technologies: Subset: setTechnologies'}
        if self.type == 'Transport':
            subsets['setAliasNodes']: 'Copy of the set of nodes to model transport. Subset: setNodes'
        return subsets

    def getTechParams(self):

        params = {
            f'minCapacity{self.type}':       f'Parameter which specifies the minimum {self.type} size that can be installed. \
                                             Dimensions: set{self.type}Technologies',
            f'maxCapacity{self.type}':       f'Parameter which specifies the maximum {self.type} size that can be installed. \
                                             Dimensions: set{self.type}Technologies',
            f'minLoad{self.type}':           f'fraction used to determine the minimum load of/ flow through the {self.type}. \
                                             Dimensions: set{self.type}Technologies',
            f'availability{self.type}':      f'node- and time-dependent availability of {self.type}. \
                                             Dimensions: set{self.type}Technologies, {self.dim}, setTimeSteps'}

        return params

    def getTechVars(self):

        variables = {
            f'install{self.type}Technologies':  f'installment of a {self.type} at node i and time t. \
                                                Dimensions: set{self.type}Technologies, {self.dim}, setTimeSteps. Domain: Binary',
            f'capacity{self.type}Technologies': f'size of {self.type} installed between nodes at time t. \
                                                Dimensions: set{self.type}Technologies, {self.dim}, setTimeSteps. Domain: NonNegativeReals'}


        return variables

    def getTechConstr(self):

        constraints = {
            f'constraint{self.type}TechnologiesMinCapacity':  f'min capacity of {self.type} technology that can be installed. \
                                                              Dimensions: set{self.type}Technologies, {self.dim}, setTimeSteps',
            f'constraint{self.type}TechnologiesMaxCapacity':  f'max capacity of {self.type} technology that can be installed. \
                                                              Dimensions: set{self.type}Technologies, {self.dim}, setTimeSteps',
            f'constraintAvailability{self.type}Technologies': f'limited availability of {self.type} technology. \
                                                              Dimensions: set{self.type}Technologies, {self.dim}, setTimeSteps'
            }

        return constraints
