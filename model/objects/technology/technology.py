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
from model.objects.element import Element

class Technology(Element):
    # empty list of elements
    listOfElements = []

    def __init__(self, object, technologyType, technology):
        """init generic technology object
        :param object: object of the abstract optimization model
        :param technologyType: type of technology that is added to the model
        :param technology: technology that is added to the model"""

        logging.info('initialize object of a generic technology')
        super().__init__(object,technology)
        self.type = technologyType
        self.tech = technology
        self.dim  = self.getDimensions()

        # set attributes
        self.minCapacity = object.pyoDict["minCapacity"][technology]
        self.maxCapacity = object.pyoDict["maxCapacity"][technology]
        self.lifetime = object.pyoDict["lifetime"][technology]
        self.availability = object.pyoDict["availability"][technology]
        # add Technology to list
        Technology.addElement(self)

        if not hasattr(self.model, f'set{self.type}Technologies'):
            sets = {f'set{self.type}Technologies': f'Set of {self.type} technologies. Subset: setTechnologies'}
            self.addSets(sets)

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

        subsets = dict()

        return subsets

    def getTechParams(self):
        """ get the parameters of the technology type
        :return params: return dictionary containing the technology parameters"""

        params = {f'minCapacity{self.tech}':  f'Parameter which specifies the minimum {self.tech} size that can be installed',
                  f'maxCapacity{self.tech}':  f'Parameter which specifies the maximum {self.tech} size that can be installed',
                #   f'maxCapacity{self.tech}':  f'Parameter which specifies the maximum {self.tech} size that can be installed',
                  f'lifetime{self.tech}':     f'Parameter which specifies the lifetime of {self.tech}.',
                  f'availability{self.tech}': f'node- and time-dependent availability of {self.tech}.\
                                              \n\t Dimensions: {self.dim}, setTimeSteps'}


        return params

    def getTechVars(self):
        """ get the variables of the technology type
        :return vars: return dictionary containing the technology variables"""

        variables = {
            f'install{self.tech}':       f'installment of a {self.tech} at node i and time t. \
                                         \n\t Dimensions: {self.dim}, setTimeSteps.\
                                         \n\t Domain: Binary',
            f'capacity{self.tech}':      f'size of {self.tech} installed at time t. \
                                         \n\t Dimensions: {self.dim}, setTimeSteps. \
                                         \n\t Domain: NonNegativeReals',
            f'buildCapacity{self.tech}': f'size of {self.tech} build at time t. \
                                         \n\t Dimensions: {self.dim}, setTimeSteps. \
                                         \n\t Domain: NonNegativeReals',
            f'capex{self.tech}':         f'capital expenditures for installing {self.tech} time t. \
                                         \n\t Dimensions: {self.dim}, setTimeSteps. \
                                         \n\t Domain: NonNegativeReals'}

        ## to enable technology expansion add the following binary variable
        # f'expand{self.tech}' which is a binary decision variable to model if the technology capacity is expanded over its lifetime'

        return variables

    def getTechConstr(self):
        """get the variables of the technology type
        :return constraints: return dictionary containing the technology constraints"""

        constraints = {f'{self.tech}Availability':     f'limited availability of {self.tech} technology. \
                                                       \n\t Dimensions: {self.dim}, setTimeSteps',
                       f'{self.tech}MinCapacity':      f'min capacity of {self.tech} technology. \
                                                       \n\t Dimensions: {self.dim}, setTimeSteps',
                       f'{self.tech}MaxCapacity':      f'max capacity of {self.tech} technology. \
                                                       \n\t Dimensions: {self.dim}, setTimeSteps',
                       f'{self.tech}Lifetime':         f'limited lifetime of {self.tech} technology. \
                                                       \n\t Dimensions: {self.dim}, setTimeSteps'}
        ## to enable technology expansion add the following constraints
        # f'{self.tech}MinCapacityExpansionRule', f'{self.tech}MaxCapacityExpansionRule', f'{self.tech}LimitCapacityExpansionRule'

        return constraints

    
