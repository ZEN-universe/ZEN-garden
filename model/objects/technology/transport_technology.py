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
from model.objects.technology.technology import Technology

class TransportTechnology(Technology):

    def __init__(self, object, tech):
        """init generic technology object
        :param object: object of the abstract optimization model"""

        logging.info('initialize object of a transport technology')
        super().__init__(object, 'Transport', tech)

        # merge new items with sets and subsets dictionary from Technology class
        subsets = {
            f'setTransportCarriers{tech}': f'carriers that can be transported with {tech}. Subset: setCarriers'}
        subsets = {**subsets, **self.getTechSubsets()}
        self.addSets(subsets)

        #%% Parameters
        params = {
            f'distance{tech}':        f'distance between two nodes for {tech}.\
                                      \n\t Dimensions: setNodes, setAliasNodes',
            f'costPerDistance{tech}': f'capex {tech} per unit distance.\
                                      \n\t Dimensions: setNodes, setAliasNodes, setTimeSteps',
            f'minFlow{tech}':         f'minimum flow through the {tech} relative to installed capacity.',
            f'maxFlow{tech}':         f'maximum flow through the {tech} relative to installed capacity.',
            f'lossFlow{tech}':        f'carrier losses due to transport with {tech}.'}

        # merge new items with parameters dictionary from Technology class
        params = {**params, **self.getTechParams()}
        self.addParams(params)

        #%% Decision variables
        variables = {
            f'carrierFlow{tech}': f'carrier flow through {tech}. \
                                  \n\t Dimensions: setTransportCarriers{tech}, setNodes, setAliasNodes, setTimeSteps.\
                                  \n\t Domain: NonNegativeReals',
            f'carrierLoss{tech}': f'carrier loss through transport with {tech}.\
                                  \n\t Dimensions: setTransportCarriers{tech}, setNodes, setAliasNodes, setTimeSteps.\
                                  \n\t Domain: NonNegativeReals',
            f'capacityAux{tech}': f'auxiliary variable of the available capacity to model min and max possible flow through {tech}. \
                                  \n\t Dimensions: setNodes, setAliasNodes, setTimeSteps.\
                                  \n\t Domain: NonNegativeReals',
            f'select{tech}':      f'binary variable to model the scheduling of {tech}.\
                                  \n\t Dimensions: setNodes, setAliasNodes, setTimeSteps.\
                                  \n\t Domain: NonNegativeReals'}
        # merge new items with variables dictionary from Technology class 
        variables = {**variables, **self.getTechVars()}
        self.addVars(variables)

        #%%  Contraints in current class
        ## add linear constraints defined for single conversion technology: set of single technology added as dimension 0
        constr = {
            f'{tech}Availability': f'limited availability of {tech} depending on node and time.\
                                   \n\t Dimensions: setNodes, setAliasNodes, setTimeSteps',
            f'{tech}MinCapacity':  f'min capacity of {tech} that can be installed.\
                                   \n\t Dimensions: setNodes, setAliasNodes, setTimeSteps',
            f'{tech}MaxCapacity':  f'max capacity of {tech} that can be installed.\
                                   \n\t Dimensions: setNodes, setAliasNodes, setTimeSteps',
            f'{tech}MinFlow':      f'min possible carrier flow through {tech}.\
                                   \n\t Dimensions: setTransportCarriers{tech}, setNodes, setAliasNodes, setTimeSteps',
            f'{tech}MaxFlow':      f'max possible carrier flow through {tech}.\
                                   \n\t Dimensions: setTransportCarriers{tech}, setNodes, setAliasNodes, setTimeSteps',
            f'{tech}AuxLBFlow':    f'LB for auxiliary variable capacityAux{tech}.\
                                   \n\t Dimensions: setTransportCarriers{tech}, setNodes, setAliasNodes, setTimeSteps',
            f'{tech}AuxUBFlow':    f'UB for auxiliary variable capacityAux{tech}.\
                                   \n\t Dimensions: setTransportCarriers{tech}, setNodes, setAliasNodes, setTimeSteps',
            f'{tech}LossesFlow':   f'Carrier loss due to transport with through {tech}.\
                                   \n\t Dimensions: setTransportCarriers{tech}, setNodes, setAliasNodes, setTimeSteps'}

        constr = {**constr, **self.getTechConstr()}
        self.addConstr(constr, replace = [tech, 'TransportTechnology'], passValues = [tech])

        logging.info(f'added subsets, parameters, decision variables and constraints for {tech}')
   
    #%% Constraint rules pre-defined in Technology class
    @staticmethod
    def constraintTransportTechnologyAvailabilityRule(model, tech, node, aliasNode, time):
        """limited availability of conversion technology"""

        # parameters
        availabilityTechnology = getattr(model, f'availability{tech}')
        # variables
        capacityTechnology     = getattr(model, f'capacity{tech}')

        return (availabilityTechnology[node, aliasNode, time]
                    >= capacityTechnology[node, aliasNode, time])

    @staticmethod
    def constraintTransportTechnologyMinCapacityRule(model, tech, node, aliasNode, time):
        """ min capacity expansion of transport technology."""

        # parameters
        minCapacityTechnology = getattr(model, f'minCapacity{tech}')
        # variables
        installTechnology  = getattr(model, f'install{tech}')
        capacityTechnology = getattr(model, f'capacity{tech}')

        return (minCapacityTechnology * installTechnology[node, aliasNode, time]
                <= capacityTechnology[node, aliasNode, time])

    @staticmethod
    def constraintTransportTechnologyMaxCapacityRule(model, tech, node, aliasNode, time):
        """max capacity expansion of transport technology"""

        # parameters
        maxCapacityTechnology = getattr(model, f'maxCapacity{tech}')
        # variables
        installTechnology  = getattr(model, f'install{tech}')
        capacityTechnology = getattr(model, f'capacity{tech}')

        return (maxCapacityTechnology * installTechnology[node, aliasNode, time]
                    >= capacityTechnology[node, aliasNode, time])

    @staticmethod
    def constraintTransportTechnologyLimitedLifetimeRule(model, tech, node, aliasNode, time):
        """limited lifetime of the technologies"""

        # parameters
        lifetime = getattr(model, f'lifetime{tech}')
        # variables
        capacityTechnology = getattr(model, f'capacity{tech}')

        # time range
        t_start = max(1, t - lifetime + 1)
        t_end = time + 1

        return (capacityTechnology[node, aliasNode, time]
                == sum((capacityTechnology[node, aliasNode, t + 1] - capacityTechnology[node, aliasNode, t]
                        for t in range(t_start, t_end))))

    @staticmethod
    def constraintTransportTechnologyMinCapacityExpansionRule(model, tech, node, aliasNode, time):
        """min capacity expansion of conversion technology"""

        # parameters
        minCapacityExpansion = getattr(model, f'minCapacityExpansion{tech}')
        # variables
        expandTechnology = getattr(model, f'expand{tech}')
        capacityTechnology = getattr(model, f'capacity{tech}')

        return (expandTechnology[node, aliasNode, t] * minCapacityExpansion
                >= capacityTechnology[node, aliasNode, time] - capacityTechnology[node, aliasNode, time])

    @staticmethod
    def constraintTransportTechnologyMaxCapacityExpansionRule(model, tech, node, aliasNode, time):
        """max capacity expansion of conversion technology"""

        # parameters
        maxCapacityExpansion = getattr(model, f'maxCapacityExpansion{tech}')
        # variables
        expandTechnology = getattr(model, f'expand{tech}')
        capacityTechnology = getattr(model, f'capacity{tech}')

        return (expandTechnology[node, aliasNode, t] * maxCapacityExpansion
                <= capacityTechnology[node, aliasNode, time] - capacityTechnology[node, aliasNode, time])

    @staticmethod
    def constraintConversionTechnologyLimitedCapacityExpansionRule(model, tech, node, aliasNode, time):
        """technology capacity can only be expanded once during its lifetime"""

        # parameters
        lifetime = getattr(model, f'lifetime{tech}')
        # variables
        installTechnology = getattr(model, f'install{tech}')

        # time range
        t_start = max(1, t - lifetime + 1)
        t_end = time + 1

        return (sum(installTechnology[node, aliasNode, t] for t in range(t_start, t_end)) <= 1)


    #%% Contraint rules defined in current class - Operation
    @staticmethod
    def constraintTransportTechnologyMaxFlowRule(model, tech, carrier, node, aliasNode, time):
        """max flow carrier through  transport technology"""

        # parameters
        maxFlowTechnology = getattr(model, f'maxFlow{tech}')
        # variables
        capacityTechnologyAux = getattr(model, f'capacityAux{tech}')
        carrierFlowTechnology = getattr(model, f'carrierFlow{tech}')

        return (carrierFlowTechnology[carrier, node, aliasNode, time]
                    <= maxFlowTechnology * capacityTechnologyAux[node, aliasNode, time])

    @staticmethod
    def constraintTransportTechnologyMinFlowRule(model, tech, carrier, node, aliasNode, time):
        """min carrier flow through transport technology"""

        # parameters
        minFlowTechnology = getattr(model, f'minFlow{tech}')
        # variables
        capacityTechnologyAux = getattr(model, f'capacity{tech}')
        carrierFlowTechnology = getattr(model, f'carrierFlow{tech}')

        return (carrierFlowTechnology[carrier, node, aliasNode, time]
                >= minFlowTechnology * capacityTechnologyAux[node, aliasNode, time])

    @staticmethod
    def constraintTransportTechnologyAuxLBFlowRule(model, tech, carrier, node, aliasNode, time):
        """coupling flow and auxiliary variable of transport technology"""

        # parameters
        maxCapacityTechnology = getattr(model, f'maxCapacity{tech}')
        # variables
        selectTechnology      = getattr(model, f'select{tech}')
        capacityTechnologyAux = getattr(model, f'capacity{tech}')
        carrierFlowTechnology = getattr(model, f'carrierFlow{tech}')

        return (carrierFlowTechnology[carrier, node, aliasNode, time]
                - maxCapacityTechnology * (1-selectTechnology[node, aliasNode, time])
                <= capacityTechnologyAux[node, aliasNode, time])

    @staticmethod
    def constraintTransportTechnologyAuxUBFlowRule(model, tech, carrier, node, aliasNode, time):
        """coupling flow and auxiliary variable of transport technology"""

        # variables
        capacityTechnologyAux = getattr(model, f'capacity{tech}')
        carrierFlowTechnology = getattr(model, f'carrierFlow{tech}')

        return (carrierFlowTechnology[carrier, node, aliasNode, time]
                <= capacityTechnologyAux[node, aliasNode, time])

    @staticmethod
    def constraintTransportTechnologyLossesFlowRule(model, tech, carrier, node, aliasNode, time):
        """compute the flow losses for a carrier through a transport technology"""

        # parameters
        distanceTransport   = getattr(model, f'distance{tech}')
        lossFactorTransport = getattr(model, f'lossFlow{tech}')
        # variables
        carrierFlowTechnology = getattr(model, f'carrierFlow{tech}')
        carrierLossTransport  = getattr(model, f'carrierLoss{tech}')

        return(carrierLossTransport[carrier, node, aliasNode, time]
               == distanceTransport[node, aliasNode] * lossFactorTransport * carrierFlowTechnology[carrier, node, aliasNode, time])

    @staticmethod
    def constraintTransportTechnologyLinearCapexRule(model, tech, node, aliasNode, time):
        """ definition of the capital expenditures for the transport technology"""

        # parameters
        distanceTransport = getattr(model, f'distance{tech}')
        costPerDistance   = getattr(model, f'costPerDistance{tech}')
        # variables
        capexTechnology    = getattr(model, f'capex{tech}')
        capacityTechnology = getattr(model, f'capacity{tech}')

        return (capexTechnology[node, aliasNode, time] == 0.5 *
                capacityTechnology[node, aliasNode, time] *
                distanceTransport[node, aliasNode] *
                costPerDistance[node, aliasNode, time])

