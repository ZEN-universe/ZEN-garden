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
from model.model_instance.objects.technology.technology import Technology

class ProductionTechnology(Technology):

    def __init__(self, object):
        """init generic technology object
        :param object: object of the abstract model"""

        logging.info('initialize object of a production technology')
        super().__init__(object, 'Production')

        #%% Sets and subsets
        subsets = {}
        if self.analysis['technologyApproximationCapex'] == 'PWA':
            #TODO currently hard-coded: each produciton technology can have a different set of support points
            subsets['setPWACapex'] = 'Set of support points for piecewise affine linearization of capex of electrolysis'
        subsets = {**subsets, **self.getTechSubsets()}
        self.addSubsets(subsets)

        #%% Parameters
        params = {'converAvailability':  'Parameter that links production technology, input, and ouput carriers. \
                                         \n\t Dimensions: setProductionTechnologies, setInputCarriers, setOutputCarriers'}
        # technology conversion efficiency
        if self.analysis['technologyApproximationEfficiency'] == 'linear':
            params['converEfficiency'] = 'Parameter which specifies the linear conversion efficiency of a technology.\
                                          \n\t Dimensions: setProductionTechnologies, setInputCarriers, setOutputCarriers'
        elif self.analysis['technologyApproximationEfficiency'] == 'PWA':
            params['slopePWAconverEfficiency'] = ''
            params['extreme0PWAconverEfficiency'] = ''
            params['extreme1PWAconverEfficiency'] = ''
            params['value0PWAconverEfficiency'] = ''

        # technology capex
        if self.analysis['technologyApproximationCapex'] == 'linear':
            params['valueCapex'] = 'Parameter which defines the coefficient of proportionality of capex. \
                                    \n\t Dimensions: setProductionTechnologies'
        elif self.analysis['technologyApproximationCapex'] == 'PWA':
            params['slopePWACapex'] = 'Parameter which specifies the slope in the PWA approximation of Capex of a production technology. \
                                          \n\t Dimensions: setProductionTechnologies, setPWACapex.'
            params['extreme0PWACapex'] = 'Parameter which specifies the 1st domain extremes in the PWA approximation of Capex of a production technology. \
                                          \n\t Dimensions: setProductionTechnologies, setPWACapex'
            params['extreme1PWACapex'] = 'Parameter which specifies the 2nd domain extremes in the PWA approximation of Capex of a production technology. \
                                          \n\t Dimensions: setProductionTechnologies, setPWACapex'
            params['value0PWACapex'] = 'Parameter which specifies the value of Capex in the PWA approximation of a production technology. \
                                          \n\t Dimensions: setProductionTechnologies, setPWACapex'
        # technology operation
            params['minLoadProduction'] = 'Parameter which specifies the minimum load of a production technology. \
                                          \n\t Dimensions: setProductionTechnologies'
            params['maxLoadProduction'] = 'Parameter which specifies the minimum load of a production technology. \
                                          \n\t Dimensions: setProductionTechnologies'

        # merge new items with parameters dictionary from Technology class
        params = {**params, **self.getTechParams()}
        self.addParams(params)

        #%% Decision variables
        variables = {
                'inputProductionTechnologies':                  'Input stream of a carrier into production technology. \
                                                                 \n\t Dimensions: setInputCarriers, setProductionTechnologies, setNodes, setTimeSteps. \
                                                                 \n\t Domain: NonNegativeReals',
                'outputProductionTechnologies':                  'Output stream of a carrier into production technology. \
                                                                 \n\t Dimensions: setOutputCarriers, setProductionTechnologies, setNodes, setTimeSteps. \
                                                                 \n\t Domain: NonNegativeReals',
                }
        # merge new items with variables dictionary from Technology class            
        variables = {**variables, **self.getTechVars()}
        self.addVars(variables)

        variables_pwa = {
            'auxiliaryPWACapex':                                'Activation of a segment in the PWA approximation of Capex. \
                                                                \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps, setPWACapex.\
                                                                \n\t Domain: Binary',
            'PWACapex':                                         'Capex per production technology. \
                                                                \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps. \
                                                                \n\t Domain: NonNegativeReals',
            'capacityAuxProductionTechnologies':                'Auxiliary variable for capacity production technology to allow PWA approximation. \
                                                                \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps, setPWACapex. \
                                                                \n\t Domain: NonNegativeReals',
            }

        if self.analysis['technologyApproximationCapex'] == 'PWA':
            self.addVars(variables_pwa)

        variables_operation = {
            'capacityAuxOpProductionTechnologies':              'Auxiliary variable to describe the operation at minimum and maximum load. \
                                                                \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps.'
                                                                '\n\t Domain: NonNegativeReals',
            'schedulingOpProductionTechnologies':               'Auxiliary variable to describe the activation of the production technology.\
                                                                \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps.'
                                                                '\n\t Domain: Binary'
            }
        self.addVars(variables_operation)

        #%% Contraints in current class
        constr = {
            # performance                                              
            'constraintProductionTechnologiesPerformance':          'Conversion efficiency of production technology. \
                                                                    \n\t Dimensions: setProductionTechnologies, setInputCarriers, setOutputCarriers, setNodes, setTimeSteps.',
            # # operation
            # 'constraintProductionTechnologiesFlowCapacity':         'Couple the flow with the installed capacity. \
            #                                                         \n \t Dimensions: setProductionTechnologies, setInputCarriers, setOutputCarriers, setNodes, setTimeSteps.'
            }

        # merge new items with constraints dictionary from Technology class                                                      
        constr = {**constr, **self.getTechConstr()}
        self.addConstr(constr)

        constr_pwa = {
            'constraintProductionTechnologiesCapex':                'Definition Capex production technology. \
                                                                    \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps.',
            'constraintProductionTechnologiesAuxiliaryPWACapex1':   'Activation of single segment PWA auxiliary variable. \
                                                                    \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps.',
            'constraintProductionTechnologiesAuxiliaryPWACapex2':   'Activation of auxiliary variable based on lower bound of PWA domain. \
                                                                    \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps.',
            'constraintProductionTechnologiesAuxiliaryPWACapex3':   'Activation of auxiliary variable based on upper bound of PWA domain. \
                                                                    \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps.',
            'constraintProductionTechnologiesAuxiliaryPWACapex4':   'Definition of auxiliary variable installation capacity with upper bound from auxiliary binary variable. \
                                                                    \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps, setPWACapex.',
            'constraintProductionTechnologiesAuxiliaryPWACapex5':   'Definition of auxiliary variable installation capacity with lower bound from auxiliary binary variable. \
                                                                    \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps, setPWACapex.',
            'constraintProductionTechnologiesAuxiliaryPWACapex6':   'Connection of auxiliary variable installation capacity with variable installation capacity. \
                                                                    \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps, setPWACapex.',
            'constraintProductionTechnologiesAuxiliaryPWACapex7':   'Connection of auxiliary variable installation capacity with variable installation capacity. \
                                                                    \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps, setPWACapex.',
            }
        if self.analysis['technologyApproximationCapex'] == 'PWA':
            self.addConstr(constr_pwa)

        constraints_operation = {
            'constraintMinProductionTechnologiesFlowCapacity':          'lower bound coupling the output energy flow and the capacity of production technology. \
                                                                        \n\t Dimensions: setProductionTechnologies, setOutputCarriers, setNodes, setTimeSteps',
            'constraintMaxProductionTechnologiesFlowCapacity':          'upper bound coupling the output energy flow and the capacity of production technology. \
                                                                        \n\t Dimensions: setProductionTechnologies, setOutputCarriers, setNodes, setTimeSteps',
            'constraintMinProductionTechnologiesAuxFlowCapacity1':      'lower bound in the definition of the auxiliary variable for the capacity of production technology. \
                                                                        \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps',
            'constraintMaxProductionTechnologiesAuxFlowCapacity1':      'upper bound in the definition of the auxiliary variable for the capacity of production technology. \
                                                                        \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps',
            'constraintMinProductionTechnologiesAuxFlowCapacity2':      'lower bound in the definition of the auxiliary variable for the capacity of production technology. \
                                                                        \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps',
            'constraintMaxProductionTechnologiesAuxFlowCapacity2':      'upper bound in the definition of the auxiliary variable for the capacity of production technology. \
                                                                        \n\t Dimensions: setProductionTechnologies, setNodes, setTimeSteps'
            }
        self.addConstr(constraints_operation)

        logging.info('added production technology sets, parameters, decision variables and constraints')

    #%% Contraint rules pre-defined in Technology class

    @staticmethod
    def constraintProductionTechnologiesMinCapacityRule(model, tech, node, time):
        """
        min size of production technology
        """
        expression = (
            model.minCapacityProduction[tech] * model.installProductionTechnologies[tech, node, time]
            <=
            model.capacityProductionTechnologies[tech, node, time]
            )
        return expression

    @staticmethod
    def constraintProductionTechnologiesMaxCapacityRule(model, tech, node, time):
        """
        max size of production technology
        """
        expression = (
            model.maxCapacityProduction[tech] * model.installProductionTechnologies[tech, node, time]
            >=
            model.capacityProductionTechnologies[tech, node, time]
            )
        return expression

    @staticmethod
    def constraintAvailabilityProductionTechnologiesRule(model, tech, node, time):
        """
        limited availability of production technology
        """
        expression = (
            model.availabilityProduction[tech, node, time]
            >=
            model.installProductionTechnologies[tech, node, time]
            )
        return expression

    #%% Contraint rules defined in current class - Efficiency

    @staticmethod
    def constraintProductionTechnologiesPerformanceRule(model, tech, carrierIn, carrierOut, node, time):
        """
        conversion efficiency of production technology
        """
        expression = None

        if model.converAvailability[tech, carrierIn, carrierOut] == 1:
            expression = (
                model.converEfficiency[tech, carrierIn, carrierOut] * model.inputProductionTechnologies[carrierIn, tech, node, time]
                ==
                model.outputProductionTechnologies[carrierOut, tech, node, time]
                )
        else:
            expression = (
                model.outputProductionTechnologies[carrierOut, tech, node, time] == 0
                )
        return expression

    #%% Contraint rules defined in current class - Design
    @staticmethod
    def constraintProductionTechnologiesCapexRule(model, tech, node, time):
        """ 
        definition of PWA capex approximation based on supporting points
        """

        expression = (
            model.PWACapex[tech, node, time]
            ==
            sum(model.slopePWACapex[tech, point] * \
            (model.capacityAuxProductionTechnologies[tech, node, time,point] - model.extreme0PWACapex[tech, point] * model.auxiliaryPWACapex[tech, node, time, point]) +\
            model.value0PWACapex[tech, point] * model.auxiliaryPWACapex[tech, node, time, point]
            for point in model.setPWACapex)
        )

        return expression

    @staticmethod
    def constraintProductionTechnologiesAuxiliaryPWACapex1Rule(model, tech, node, time):
        """
        constraint on activation of single segment capex approximation
        """
        expression = (
            sum(model.auxiliaryPWACapex[tech, node, time, point] for point in model.setPWACapex)
            ==
            1)
        return expression

    @staticmethod
    def constraintProductionTechnologiesAuxiliaryPWACapex2Rule(model, tech, node, time):
        """
        activation of auxiliary variable based on lower bound of PWA domain
        """
        expression = (
            model.capacityProductionTechnologies[tech, node, time]
            >=
            sum(model.auxiliaryPWACapex[tech, node, time, point] * model.extreme0PWACapex[tech, point] for point in model.setPWACapex)
            )

        return expression

    @staticmethod
    def constraintProductionTechnologiesAuxiliaryPWACapex3Rule(model, tech, node, time):
        """
        activation of auxiliary variable based on upper bound of PWA domain
        """
        expression = (
            model.capacityProductionTechnologies[tech, node, time]
            <=
            sum(model.auxiliaryPWACapex[tech, node, time, point] * model.extreme1PWACapex[tech, point] for point in model.setPWACapex)
            )

        return expression

    @staticmethod
    def constraintProductionTechnologiesAuxiliaryPWACapex4Rule(model, tech, node, time, point):
        """
        definition of auxiliary variable installation capacity with upper bound from auxiliary binary variable
        """
        expression = (
            model.capacityAuxProductionTechnologies[tech, node, time, point]
            <=
            model.maxCapacityProduction[tech] * model.auxiliaryPWACapex[tech, node, time, point]
            )
        return expression

    @staticmethod
    def constraintProductionTechnologiesAuxiliaryPWACapex5Rule(model, tech, node, time, point):
        """
        definition of auxiliary variable installation capacity with lower bound from auxiliary binary variable
        """
        expression = (
            model.capacityAuxProductionTechnologies[tech, node, time, point]
            >=
            model.minCapacityProduction[tech] * model.auxiliaryPWACapex[tech, node, time, point]
            )
        return expression

    @staticmethod
    def constraintProductionTechnologiesAuxiliaryPWACapex6Rule(model, tech, node, time,point):
        """
        connection of auxiliary variable installation capacity with variable installation capacity
        """
        expression = (
            model.capacityAuxProductionTechnologies[tech, node, time, point]
            <=
            model.capacityProductionTechnologies[tech, node, time]
            )
        return expression

    @staticmethod
    def constraintProductionTechnologiesAuxiliaryPWACapex7Rule(model, tech, node, time, point):
        """
        connection of auxiliary variable installation capacity with variable installation capacity
        """
        expression = (
            model.capacityAuxProductionTechnologies[tech, node, time, point]
            >=
            model.capacityProductionTechnologies[tech, node, time] - \
                model.maxCapacityProduction[tech] * (1-model.auxiliaryPWACapex[tech, node, time, point])
            )
        return expression

    #TODO implement conditioning for e.g. hydrogen

    #%% Contraint rules defined in current class - Operation

    # @staticmethod
    # def constraintProductionTechnologiesFlowCapacityRule(model, tech, carrierIn, carrierOut, node, time):
    #     """
    #     coupling the output energy flow and the capacity of production technology
    #     """
    #     expression = None
    #
    #     if model.converAvailability[tech, carrierIn, carrierOut] == 1:
    #         expression = (
    #             model.capacityProductionTechnologies[tech, node, time]
    #             ==
    #             model.outputProductionTechnologies[carrierOut, tech, node, time]
    #             )
    #     else:
    #         expression = (
    #             model.outputProductionTechnologies[carrierOut, tech, node, time] == 0
    #             )
    #     return expression

    @staticmethod
    def constraintMinProductionTechnologiesFlowCapacityRule(model, tech, carrierOut, node, time):
        """
        lower bound coupling the output energy flow and the capacity of production technology
        """
        expression = (
            model.capacityAuxOpProductionTechnologies[tech, node, time]*model.minLoadProduction[tech]
            <=
            model.outputProductionTechnologies[carrierOut, tech, node, time]
            )
        return expression

    @staticmethod
    def constraintMaxProductionTechnologiesFlowCapacityRule(model, tech, carrierOut, node, time):
        """
        upper bound coupling the output energy flow and the capacity of production technology
        """
        expression = (
                model.capacityAuxOpProductionTechnologies[tech, node, time] * model.maxLoadProduction[tech]
                >=
                model.outputProductionTechnologies[carrierOut, tech, node, time]
        )
        return expression

    @staticmethod
    def constraintMinProductionTechnologiesAuxFlowCapacity1Rule(model, tech, node, time):
        """
        lower bound in the definition of the auxiliary variable for the capacity of production technology
        """
        expression = (
                model.capacityProductionTechnologies[tech, node, time] - model.maxCapacityProduction[tech]*(1-model.schedulingOpProductionTechnologies[tech, node, time])
                <=
                model.capacityAuxOpProductionTechnologies[tech, node, time]
        )
        return expression

    @staticmethod
    def constraintMaxProductionTechnologiesAuxFlowCapacity1Rule(model, tech, node, time):
        """
        upper bound in the definition of the auxiliary variable for the capacity of production technology
        """
        expression = (
                model.capacityProductionTechnologies[tech, node, time]
                >=
                model.capacityAuxOpProductionTechnologies[tech, node, time]
        )
        return expression

    @staticmethod
    def constraintMinProductionTechnologiesAuxFlowCapacity2Rule(model, tech, node, time):
        """
        lower bound in the definition of the auxiliary variable for the capacity of production technology
        """
        expression = (
                model.minCapacityProduction[tech]*model.schedulingOpProductionTechnologies[tech,node,time]
                <=
                model.capacityAuxOpProductionTechnologies[tech, node, time]
        )
        return expression

    @staticmethod
    def constraintMaxProductionTechnologiesAuxFlowCapacity2Rule(model, tech, node, time):
        """
        upper bound in the definition of the auxiliary variable for the capacity of production technology
        """
        expression = (
                model.maxCapacityProduction[tech] * model.schedulingOpProductionTechnologies[tech, node, time]
                >=
                model.capacityAuxOpProductionTechnologies[tech, node, time]
        )
        return expression