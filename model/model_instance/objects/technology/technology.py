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
from model.model_instance.objects.element import Element

class Technology(Element):

    def __init__(self, object, technologyType):
        """init generic technology object
        :param object: object of the abstract optimization model
        :param technologyType: type of technology that is added to the model"""

        logging.info('initialize object of a generic technology')
        super().__init__(object)
        self.type = technologyType
        self.dim  = self.getDimensions()

    def getDimensions(self):
        """ determine dimensions depending on the technology type
        :return dim: return dimensions"""

        if self.type == 'Transport':
            dim = 'setNodes, setAliasNodes'
        else:
            dim = 'setNodes'

        return dim

    def getTechSubsets(self):
        """ get the subsets of the technology type
        :return subsets: return dictionary containing the technology subsets"""

        subsets = {
            f'set{self.type}Technologies': f'Set of {self.type} technologies: Subset: setTechnologies'}
        if self.type == 'Transport':
            subsets['setAliasNodes']: 'Copy of the set of nodes to model transport. Subset: setNodes'
        return subsets

    def getTechParams(self):
        """ get the parameters of the technology type
        :return params: return dictionary containing the technology parameters"""

        params = {
            f'minCapacity{self.type}':              f'Parameter which specifies the minimum {self.type} size that can be installed. \
                                                    \n\t Dimensions: set{self.type}Technologies',
            f'maxCapacity{self.type}':              f'Parameter which specifies the maximum {self.type} size that can be installed. \
                                                    \n\t Dimensions: set{self.type}Technologies',
            f'availability{self.type}':             f'node- and time-dependent availability of {self.type}. \
                                                    \n\t Dimensions: set{self.type}Technologies, {self.dim}, setTimeSteps.'                                             
            # f'minLoad{self.type}':           f'fraction used to determine the minimum load of/ flow through the {self.type}. \
            #                                  \n\t Dimensions: set{self.type}Technologies',
            }

        return params

    def getTechVars(self):
        """ get the variables of the technology type
        :return vars: return dictionary containing the technology variables"""

        variables = {
            f'install{self.type}Technologies':      f'installment of a {self.type} at node i and time t. \
                                                    \n\t Dimensions: set{self.type}Technologies, {self.dim}, setTimeSteps.\
                                                    \n\t Domain: Binary',
            f'capacity{self.type}Technologies':     f'size of {self.type} installed between nodes at time t. \
                                                    \n\t Dimensions: set{self.type}Technologies, {self.dim}, setTimeSteps. \
                                                    \n\t Domain: NonNegativeReals'
            }

        return variables

    def getTechConstr(self):
        """get the variables of the technology type
        :return constraints: return dictionary containing the technology constraints"""

        constraints = {
            f'constraint{self.type}TechnologiesMinCapacity':  f'min capacity of {self.type} technology that can be installed. \
                                                              \n\t Dimensions: set{self.type}Technologies, {self.dim}, setTimeSteps',
            f'constraint{self.type}TechnologiesMaxCapacity':  f'max capacity of {self.type} technology that can be installed. \
                                                              \n\t Dimensions: set{self.type}Technologies, {self.dim}, setTimeSteps',
            f'constraintAvailability{self.type}Technologies': f'limited availability of {self.type} technology. \
                                                              \n\t Dimensions: set{self.type}Technologies, {self.dim}, setTimeSteps'
            }

        return constraints
