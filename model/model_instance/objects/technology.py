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

    def __init__(self, object, type):
        """init generic technology object"""

        logging.info('initialize object of a generic technology')
        super().__init__(object)
        self.type = type
        self.dim  = self.getDimensions()

    def getDimensions(self):

        if self.type == 'Transport':
            dim = 'setNodes, setAliasNodes'
        else:
            dim = 'setNodes'

        return dim

    def getTechSubsets(self):

        subsets = {
            f'set{self.type}': f'Set of {self.type} technologies: Subset: setTechnologies'}
        if self.type == 'Transport':
            subsets['setAliasNodes']: 'Copy of the set of nodes to model transport. Subset: setNodes'
        return subsets

    def getTechParams(self):

        params = {
            f'sizeMin{self.type}Tech':       f'Parameter which specifies the minimum {self.type} size that can be installed. \
                                             Dimensions: set{self.type}',
            f'sizeMax{self.type}Tech':       f'Parameter which specifies the maximum {self.type} size that can be installed. \
                                             Dimensions: set{self.type}',
            f'minLoad{self.type}Tech':       f'fraction used to determine the minimum load of/ flow through the {self.type}. \
                                             Dimensions: set{self.type}',
            f'availability{self.type}Tech':  f'node- and time-dependent availability of {self.type}. \
                                             Dimensions: set{self.type}, {self.dim}, setTimeSteps'}

        return params

    def getTechVars(self):

        vars = {
            f'install{self.type}Tech': f'installment of a {self.type} at node i and time t. \
                                       Dimensions: set{self.type}, {self.dim}, setTimeSteps. Domain: Binary',
            f'size{self.type}Tech':    f'size of {self.type} installed between nodes at time t. \
                                       Dimensions: set{self.type}, {self.dim}, setTimeSteps. Domain: NonNegativeReals'}

        if type != 'Transport':
            vars[f'input{self.type}Tech']   = f'Input stream of a carrier into {self.type}. \
                                                Dimensions: setCarriersIn, set{self.type}, setNodes, setTimeSteps. Domain: NonNegativeReals'
            vars[f'output{self.type}Tech']  = f'Output stream of a carrier into {self.type}. \
                                                Dimensions: setCarriersOut, set{self.type}, setNodes, setTimeSteps. Domain: NonNegativeReals'

        return vars

    def getTechConstr(self):

        constraints = {
            f'constraint{self.type}TechSize':         f'size restriction of {self.type} technology that can be installed. \
                                                      Dimensions: set{self.type}, {self.dim}, setTimeSteps',
            f'constraintMinLoad{self.type}Tech1':     f'min load of {self.type} technology, part one. \
                                                      Dimensions: setCarriers, set{self.type}, {self.dim}, setTimeSteps',
            f'constraintMinLoad{self.type}Tech2':     f'min load of {self.type} technology, part two. \
                                                      Dimensions: setCarriers, set{self.type}, {self.dim}, setTimeSteps',
            f'constraintMaxLoad{self.type}Tech':      f'max load of {self.type} technology. \
                                                      Dimensions: setCarriers, set{self.type}, {self.dim}, setTimeSteps',
            f'constraintAvailability{self.type}Tech': f'limited availability of {self.type} technology. \
                                                      Dimensions: set{self.type}, {self.dim}, setTimeSteps'}

        return constraints
    #'def SizeRule(model, tech, node, time): """min and max size of {tech}. Dimensions: set{tech}, dim, setTimeSteps""" \
    #        return (model.maxSize{tech}[tech] * model.install{tech}[tech, node, time], \
    #            model.size{tech}[tech, node], \
    #            model.maxSize{tech}[tech] * model.install{tech}[tech, node, time])'
