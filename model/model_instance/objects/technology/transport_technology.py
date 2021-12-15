"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining the parameters, variables and constraints that hold for all transport technologies.
              The class takes the abstract optimization model as an input, and returns the parameters, variables and
              constraints that hold for the transport technologies.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
from model.model_instance.objects.technology.technology import Technology

class TransportTechnology(Technology):

    def __init__(self, object):
        """init generic technology object
        :param object: object of the abstract optimization model"""

        logging.info('initialize object of a transport technology')
        super().__init__(object, 'Transport')

        #%% Sets and subsets
        subsets = {
            'setAliasNodes': 'Copy of the set of nodes to model transport. Subset: setNodes'}
        # merge new items with sets and subsets dictionary from Technology class  
        subsets = {**subsets, **self.getTechSubsets()}
        self.addSubsets(subsets)

        #%% Parameters
        params = {
            'distanceEucledian':        'eucledian distance between any input node. \
                                         \n\t Dimensions: setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps.',
            'costPerDistance':          'capex tranport technology per unit distance. \
                                         \n\t Dimensions: setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps.',
            # 'minTransportTechLoad':     'fraction of installed transport technology size that determines the minimum load of the transport technology. \
            #                              \n\t Dimensions: setTransportTechnologies'            
            }
        # merge new items with parameters dictionary from Technology class 
        params = {**params, **self.getTechParams()}
        self.addParams(params)

        params_operation = {}
        params_operation['minLoadTransport'] = 'Parameter which specifies the minimum load of a transport technology. \
                                      \n\t Dimensions: setTransportTechnologies'
        params_operation['maxLoadTransport'] = 'Parameter which specifies the minimum load of a transport technology. \
                                      \n\t Dimensions: setTransportTechnologies'
        self.addParams(params_operation)

        #%% Decision variables
        variables = {
            'carrierFlow':            'carrier flow through transport technology from node i to node j. \
                                       \n\t Dimensions: setTransportCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps.\
                                       \n\t Domain: NonNegativeReals',
            }
        # merge new items with variables dictionary from Technology class 
        variables = {**variables, **self.getTechVars()}
        self.addVars(variables)

        variables_operation = {
            'capacityAuxOpTransportTechnologies':              'Auxiliary variable to describe the operation at minimum and maximum load. \
                                                                \n\t Dimensions: setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps.'
                                                                '\n\t Domain: NonNegativeReals',
            'schedulingOpTransportTechnologies':               'Auxiliary variable to describe the activation of the transport technology.\
                                                                \n\t Dimensions: setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps.'
                                                                '\n\t Domain: Binary'
            }
        self.addVars(variables_operation)

        #%%  Contraints in current class 
        constr = {
                'constraintTransportTechnologiesFlowCapacity':     'coupling flow carrier to capcity transport technology.\
                                                                    \n\t Dimensions: setTransportCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps.' 
                  # 'constraintTransportTechnologiesPerformance':  'performance of transport technology. \
                  #                                               \n\t Dimensions: setTransportCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps',
                  }
        # TODO add constraints for transport losses
        constr = {**constr, **self.getTechConstr()}
        self.addConstr(constr)

        constraints_operation = {
            'constraintMinTransportTechnologiesFlowCapacity':          'lower bound coupling the energy flow and the capacity of trasnport technology. \
                                                                        \n\t Dimensions: setTransportCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps',
            'constraintMaxTransportTechnologiesFlowCapacity':          'upper bound coupling the energy flow and the capacity of trasnport technology. \
                                                                        \n\t Dimensions: setTransportCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps',
            'constraintMinTransportTechnologiesAuxFlowCapacity1':      'lower bound in the definition of the auxiliary variable for the capacity of trasnport technology. \
                                                                        \n\t Dimensions: setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps',
            'constraintMaxTransportTechnologiesAuxFlowCapacity1':      'upper bound in the definition of the auxiliary variable for the capacity of transport technology. \
                                                                        \n\t Dimensions: setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps',
            'constraintMinTransportTechnologiesAuxFlowCapacity2':      'lower bound in the definition of the auxiliary variable for the capacity of transport technology. \
                                                                        \n\t Dimensions: setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps',
            'constraintMaxTransportTechnologiesAuxFlowCapacity2':      'upper bound in the definition of the auxiliary variable for the capacity of transport technology. \
                                                                        \n\t Dimensions: setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps'
            }
        self.addConstr(constraints_operation)

        logging.info('added transport technology sets, parameters, decision variables and constraints')
   
    #%% Contraint rules pre-defined in Technology class
    @staticmethod
    def constraintTransportTechnologiesMinCapacityRule(model, tech, node, aliasNode, time):
        """
        min size of transport technology.
        """
        expression = (
            model.minCapacityTransport[tech] * model.installTransportTechnologies[tech, node, aliasNode, time]
            <= 
            model.capacityTransportTechnologies[tech, node, aliasNode, time]
            )
        
        return expression
    
    @staticmethod
    def constraintTransportTechnologiesMaxCapacityRule(model, tech, node, aliasNode, time):
        """
        max size of transport technology.
        """
        expression = (
            model.maxCapacityTransport[tech] * model.installTransportTechnologies[tech, node, aliasNode, time]
            >=
            model.capacityTransportTechnologies[tech, node, aliasNode, time]
            )
        return expression 
    
    @staticmethod
    def constraintAvailabilityTransportTechnologiesRule(model, tech, node, aliasNode, time):
        """
        limited availability of transport technology.
        """
        expression = (
            model.availabilityTransport[tech, node, aliasNode, time]
            >= 
            model.installTransportTechnologies[tech, node, aliasNode, time]
            )
        return expression

    #%% Contraint rules defined in current class - Operation

    @staticmethod
    def constraintTransportTechnologiesFlowCapacityRule(model, carrier, tech, node, nodeAlias, time):
        """
        coupling flow carrier to capcity transport technology
        """
        
        expression = (
            model.capacityTransportTechnologies[tech, node, nodeAlias, time] 
            ==
            model.carrierFlow[carrier, tech, node, nodeAlias, time]
            )
        return expression
    # Rules in current class
    # @staticmethod
    # def constraintTransportTechnologiesPerformanceRule(model, carrier, tech, node, aliasNode, time):
    #     """constraint to account for transport losses.
    #     \n\t Dimensions: setTransportTechnologies, setTransportCarriers, setNodes, setTimeSteps"""
    #     # TODO implement transport losses
    #     return (model.carrierFlow[carrier, tech, node, aliasNode, time]
    #             == model.carrierFlow[carrier,tech, aliasNode, node, time])

    #%% Constraint rules defined in current class - Operation

    @staticmethod
    def constraintMinTransportTechnologiesFlowCapacityRule(model, carrier, tech, node, aliasNode, time):
        """
        lower bound coupling the output energy flow and the capacity of transport technology
        """
        expression = (
            model.capacityAuxOpTransportTechnologies[tech, node, aliasNode, time]*model.minLoadTransport[tech]
            <=
            model.carrierFlow[carrier, tech, node, aliasNode, time]
            )
        return expression

    @staticmethod
    def constraintMaxTransportTechnologiesFlowCapacityRule(model, carrier, tech, node, aliasNode, time):
        """
        upper bound coupling the output energy flow and the capacity of transport technology
        """
        expression = (
            model.capacityAuxOpTransportTechnologies[tech, node, aliasNode, time]*model.maxLoadTransport[tech]
            >=
            model.carrierFlow[carrier, tech, node, aliasNode, time]
            )
        return expression

    @staticmethod
    def constraintMinTransportTechnologiesAuxFlowCapacity1Rule(model, tech, node, aliasNode, time):
        """
        lower bound in the definition of the auxiliary variable for the capacity of transport technology
        """
        expression = (
                model.capacityTransportTechnologies[tech, node, aliasNode, time] - model.maxCapacityTransport[tech]*(1-model.schedulingOpTransportTechnologies[tech, node, aliasNode, time])
                <=
                model.capacityAuxOpTransportTechnologies[tech, node, aliasNode, time]
        )
        return expression

    @staticmethod
    def constraintMaxTransportTechnologiesAuxFlowCapacity1Rule(model, tech, node, aliasNode, time):
        """
        upper bound in the definition of the auxiliary variable for the capacity of transport technology
        """
        expression = (
                model.capacityTransportTechnologies[tech, node, aliasNode, time]
                >=
                model.capacityAuxOpTransportTechnologies[tech, node, aliasNode, time]
        )
        return expression

    @staticmethod
    def constraintMinTransportTechnologiesAuxFlowCapacity2Rule(model, tech, node, aliasNode, time):
        """
        lower bound in the definition of the auxiliary variable for the capacity of transport technology
        """
        expression = (
                model.minCapacityTransport[tech]*model.schedulingOpTransportTechnologies[tech, node, aliasNode, time]
                <=
                model.capacityAuxOpTransportTechnologies[tech, node, aliasNode, time]
        )
        return expression

    @staticmethod
    def constraintMaxTransportTechnologiesAuxFlowCapacity2Rule(model, tech, node, aliasNode, time):
        """
        upper bound in the definition of the auxiliary variable for the capacity of transport technology
        """
        expression = (
                model.maxCapacityTransport[tech]*model.schedulingOpTransportTechnologies[tech, node, aliasNode, time]
                >=
                model.capacityAuxOpTransportTechnologies[tech, node, aliasNode, time]
        )
        return expression