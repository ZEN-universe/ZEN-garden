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

    def getDimensions(tech):

        if tech == 'TransportTech':
            dim = 'setNodes, setAliasNodes'
        else:
            dim = 'setNodes'

        return dim

    def getTechParams(self,tech):

        dim = self.getDimensions(tech)
        params = {
            f'sizeMin{tech}':       f'Parameter which specifies the minimum {tech} size that can be installed. Dimensions: set{tech}',
            f'sizeMax{tech}':       f'Parameter which specifies the maximum {tech} size that can be installed. Dimensions: set{tech}',
            f'minLoad{tech}':       f'fraction used to determine the minimum load of/ flow through the {tech}. Dimensions: set{tech}',
            f'availability{tech}':  f'node- and time-dependent availability of {tech}. Dimensions: set{tech}, {dim}, setTimeSteps'}
        return params

    def getTechVars(self,tech):

        dim = self.getDimensions(tech)
        vars = {
            f'install{tech}': f'installment of a {tech} at node i and time t. Dimensions: set{tech}, {dim}, setTimeSteps. Domain: Binary',
            f'size{tech}':    f'size of {tech} installed between nodes at time t. Dimensions: set{tech}, {dim}, setTimeSteps. Domain: NonNegativeReals'}

        if tech not 'TransportTech':
            vars[f'input{tech}']   = f'Input stream of a carrier into {tech}. Dimensions: setCarrierIn, set{tech}, setNodes, setTimeSteps',
            vars[f'output{tech}']  = f'Output stream of a carrier into {tech}. Dimensions: setCarrierOut, set{tech}, setNodes, setTimeSteps'

        return vars

    def getConstraints(self, tech):

        dim = self.getDimensions(tech)
        constraints = {
            f'constraint{tech}Size':         f'size restriction of {tech} that can be installed. set{tech}, {dim}, setTimeSteps',
            f'constraintMinLoad{tech}1':     f'min load of {tech}, part one. Dimensions: setCarrier, set{tech}, {dim}, setTimeSteps',
            f'constraintMinLoad{tech}2':     f'min load of {tech}, part two. Dimensions: setCarrier, set{tech}, {dim}, setTimeSteps',
            f'constraintMaxLoad{tech}':      f'max load of {tech}. Dimensions: , setTransportTech, set{tech}, {dim}, setTimeSteps',
            f'constraintAvailability{tech}': f'limited availability of {tech}. Dimensions: set{tech}, {dim}, setTimeSteps'}

        return constraints

    #'def SizeRule(model, tech, node, time): """min and max size of {tech}. Dimensions: set{tech}, dim, setTimeSteps""" \
    #        return (model.maxSize{tech}[tech] * model.install{tech}[tech, node, time], \
    #            model.size{tech}[tech, node], \
    #            model.maxSize{tech}[tech] * model.install{tech}[tech, node, time])'
