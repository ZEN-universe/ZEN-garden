"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining the parameters, variables and constraints of the production technologies.
              The class takes the abstract optimization model as an input, and adds parameters, variables and
              constraints of the production technologies.
==========================================================================================================================================================================="""

import logging
import pyomo.environ as pe
from model.model_instance.objects.technology.technology import Technology

class ProductionTechnology(Technology):

    def __init__(self, object):
        """init generic technology object
        :param object: object of the abstract model"""

        logging.info('initialize object of a production technology')
        super().__init__(object, 'Production')

        # SETS AND SUBSETS
        subsets = {}
        if self.analysis['technologyApproximation'] == 'PWA':
            self.subsets['setSupportPointsPWA'] = 'Set of support points for piecewise affine linearization'
        subsets = {**subsets, **self.getTechSubsets()}
        self.addSubsets(subsets)

        # PARAMETERS
        params = {'converAvailability':  'Parameter that links production technology, input, and ouput carriers. \
                                         \n\t Dimensions: setProductionTechnologies, setInputCarriers, setOutputCarriers'}
        if self.analysis['technologyApproximation'] == 'linear':
            params['converEfficiency'] = 'Parameter which specifies the linear conversion efficiency of a technology.\
                                          \n\t Dimensions: setProductionTechnologies, setInputCarriers, setOutputCarriers'
        elif self.analysis['technologyApproximation'] == 'PWA':
            params['converEfficiency'] = 'Parameter which specifies the linear conversion efficiency of a technology. \
                                          \n\t Dimensions: setProductionTechnologies, setInputCarriers, setOutputCarriers, setSupportPointsPWA'
        params = {**params, **self.getTechParams()}
        self.addParams(params)

        # DECISION VARIABLES
        vars = {'inputProductionTechnologies':     'Input stream of a carrier into production technology. \
                                                    \n\t Dimensions: setInputCarriers, setProductionTechnologies, setNodes, setTimeSteps. \n\t Domain: NonNegativeReals',
                'outputProductionTechnologies':    'Output stream of a carrier into production technology. \
                                                    \n\t Dimensions: setOutputCarriers, setProductionTechnologies, setNodes, setTimeSteps. \n\t Domain: NonNegativeReals',
                'outputProductionTechnologiesAux': 'Auxiliary variable to describe output stream of a carrier into production technology. \
                                                    \n\t Dimensions: setOutputCarriers, setProductionTechnologies, setNodes, setTimeSteps. \n\t Domain: NonNegativeReals'
                }
        vars = {**vars, **self.getTechVars()}
        self.addVars(vars)
        #TODO implement conditioning for e.g. hydrogen
        #'converEnergy': 'energy involved in conversion of carrier. \n\t Dimensions: setCarriers, setNodes, setTimeSteps. \n\t Domain: NonNegativeReals'

        # CONSTRAINTS
        constr = {
            'constraintProductionTechnologiesPerformance': 'Conversion efficiency of production technology. \
                                                            \n\t Dimensions: setProductionTechnologies, setInputCarriers, setOutputCarriers, setNodes, setTimeSteps',
            'constraintMinLoadProductionTechnologies1':    'min load of production technology, part one. \
                                                            \n\t Dimensions: setOutputCarriers, setProductionTechnologies, setNodes, setTimeSteps',
            'constraintMinLoadProductionTechnologies2':    'min load of production technology, part two. \
                                                            \n\t Dimensions: setOutputCarriers, setProductionTechnologies, setNodes, setTimeSteps',
            'constraintMaxLoadProductionTechnologies1':    'max load of production technology, part one. \
                                                            \n\t Dimensions: setOutputCarriers, setProductionTechnologies, setNodes, setTimeSteps',
            'constraintMaxLoadProductionTechnologies2':    'max load of production technology, part two. \
                                                            \n\t Dimensions: setOutputCarriers, setProductionTechnologies, setNodes, setTimeSteps'
                                                            }
        constr = {**constr, **self.getTechConstr()}
        self.addConstr(constr)

        logging.info('added production technology sets, parameters, decision variables and constraints')

    # RULES
    @staticmethod
    def constraintProductionTechnologiesPerformanceRule(model, tech, carrierIn, carrierOut, node, time):
        """conversion efficiency of production technology.
        \n\t Dimensions: setProductionTechnologies, setInputCarriers, setNodes, setTimeSteps"""
    
        if model.converAvailability[tech, carrierIn, carrierOut]==1:
            return(model.converEfficiency[tech, carrierIn, carrierOut]
                   * model.inputProductionTechnologies[carrierIn, tech, node, time]
                    <= model.outputProductionTechnologies[carrierOut, tech, node, time])
        else:
            return(model.outputProductionTechnologies[carrierOut, tech, node, time] == 0)

    # Rules that are pre-defined in Technology class
    # Capacity constraints
    @staticmethod
    def constraintProductionTechnologiesMinCapacityRule(model, tech, node, time):
        """min size of production technology.
         \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps"""
    
        return (model.minCapacityProduction[tech] * model.installProductionTechnologies[tech, node, time]
                <= model.capacityProductionTechnologies[tech, node, time])

    @ staticmethod
    def constraintProductionTechnologiesMaxCapacityRule(model, tech, node, time):
        """max size of production technology.
        \n\t Dimensions: setProductionTechnologiesnologies, setNodes, setTimeSteps"""

        return (model.maxCapacityProduction[tech] * model.installProductionTechnologies[tech, node, time]
                >= model.capacityProductionTechnologies[tech, node, time])

    # Operational constraints
    @staticmethod
    def constraintMinLoadProductionTechnologies1Rule(model, carrier, tech, node, time):
        """min amount of carrier produced with production technology.
        \n\t Dimensions: setCarrier, setProductionTechnologies, setNodes, setTimeSteps"""

        return (model.minLoadProduction[tech] *  model.minCapacityProduction[tech]
                * model.installProductionTechnologies[tech, node, time]
                <= model.outputProductionTechnologiesAux[carrier, tech, node, time])

    @staticmethod
    def constraintMinLoadProductionTechnologies2Rule(model, carrier, tech, node, time):
        """min amount of carrier produced with production technology between two nodes.
        \n\t Dimensions: setCarrier, setProductionTechnologies, setNodes, setTimeSteps"""

        return (model.outputProductionTechnologies[carrier, tech, node, time]
                - model.maxCapacityProduction[tech] * (1 - model.installProductionTechnologies[tech, node, time])
                <= model.outputProductionTechnologiesAux[carrier, tech, node, time])

    @staticmethod
    def constraintMaxLoadProductionTechnologies1Rule(model, carrier, tech, node, time):
        """max amount of carrier produced with production technology.
        \n\t Dimensions: setCarrier, setProductionTechnologies, setNodes, setTimeSteps"""

        return (model.capacityProductionTechnologies[tech, node, time]
                >= model.outputProductionTechnologiesAux[carrier, tech, node, time])

    @staticmethod
    def constraintMaxLoadProductionTechnologies2Rule(model, carrier, tech, node, time):
        """max amount of carrier produced with production technology.
        \n\t Dimensions: setCarrier, setProductionTechnologies, setNodes, setTimeSteps"""

        return (model.outputProductionTechnologies[carrier, tech, node, time]
                >= model.outputProductionTechnologiesAux[carrier, tech, node, time])

    @staticmethod
    def constraintAvailabilityProductionTechnologiesRule(model, tech, node, time):
        """limited availability of production technology.
        \n\t Dimensions: setProductionTechnologiesnologies, setNodes, setTimeSteps"""

        return (model.availabilityProduction[tech, node, time]
                <= model.installProductionTechnologies[tech, node, time])