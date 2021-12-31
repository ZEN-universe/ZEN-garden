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
            f'distanceEucledian{tech}': f'eucledian distance between two nodes for {tech}.\
                                        \n\t Dimensions: setNodes, setAliasNodes, setTimeSteps',
            f'costPerDistance{tech}':   f'capex {tech} per unit distance.\
                                        \n\t Dimensions: setNodes, setAliasNodes, setTimeSteps',
            # 'minTransportTechLoad':     'fraction of installed transport technology size that determines the minimum load of the transport technology. \
            #                              \n\t Dimensions: setTransportTechnologies'
            }
        # merge new items with parameters dictionary from Technology class 
        params = {**params, **self.getTechParams()}
        self.addParams(params)

        #%% Decision variables
        variables = {
            f'capexTransportTechnology':   f'capex of transport capacity used in definition of objective function. \
                                       \n\t Dimensions: setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps.\
                                       \n\t Domain: NonNegativeReals',
            f'carrierFlow{tech}':      f'carrier flow through transport technology from node i to node j. \
                                       \n\t Dimensions: setTransportCarriers{tech}, setNodes, setAliasNodes, setTimeSteps.\
                                       \n\t Domain: NonNegativeReals',
            # 'carrierFlowAux':         'auxiliary variable to model the min possible flow through transport technology from node i to node j. \
            #                            \n\t Dimensions: setTransportCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps.\
            #                            \n\t Domain: NonNegativeReals'
            }
        # merge new items with variables dictionary from Technology class 
        variables = {**variables, **self.getTechVars()}
        self.addVars(variables)

        #%%  Contraints in current class

        ## add linear constraints defined for all the conversion technologies
        constr = {
            f'TransportTechnologyLinearCapexValue': f'Definition of Capex for all the transport technologies.\
                                                    \n\t Dimensions: setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps.'}
        self.addConstr(constr)

        ## add linear constraints defined for single conversion technology: set of single technology added as dimension 0
        constr = {
            f'{tech}Availability': f'limited availability of {tech} depending on node and time.\
                                   \n\t Dimensions: setNodes, setAliasNodes, setTimeSteps.',
            f'{tech}MinCapacity':  f'min capacity of {tech} that can be installed.\
                                   \n\t Dimensions: setNodes, setAliasNodes, setTimeSteps.',
            f'{tech}MaxCapacity':  f'max capacity of {tech} that can be installed.\
                                   \n\t Dimensions: setNodes, setAliasNodes, setTimeSteps.',
            f'{tech}FlowCapacity': f'carrier flow through {tech}.\
                                   \n\t Dimensions: setTransportCarriers{tech}, setNodes, setAliasNodes, setTimeSteps.'}
            # TODO add constraints for transport losses
        constr = {**constr, **self.getTechConstr()}
        self.addConstr(constr, replace = [tech, 'TransportTechnology'], passValues = [tech])

        logging.info(f'added subsets, parameters, decision variables and constraints for {tech}')
   
    #%% Constraint rules pre-defined in Technology class
    @staticmethod
    def constraintTransportTechnologyAvailabilityRule(model, tech, node, aliasNode, time):
        """limited availability of production technology"""

        # parameters
        availabilityTechnology = getattr(model, f'availability{tech}')
        # variables
        installTechnology = getattr(model, f'install{tech}')

        return (availabilityTechnology[node, aliasNode, time]
                    >= installTechnology[node, aliasNode, time])

    @staticmethod
    def constraintTransportTechnologyMinCapacityRule(model, tech, node, aliasNode, time):
        """ min size of transport technology."""

        # parameters
        minCapacityTechnology = getattr(model, f'minCapacity{tech}')
        # variables
        installTechnology  = getattr(model, f'install{tech}')
        capacityTechnology = getattr(model, f'capacity{tech}')

        return (minCapacityTechnology * installTechnology[node, aliasNode, time]
                <= capacityTechnology[node, aliasNode, time])

    @staticmethod
    def constraintTransportTechnologyMaxCapacityRule(model, tech, node, aliasNode, time):
        """max size of transport technology"""

        # parameters
        maxCapacityTechnology = getattr(model, f'maxCapacity{tech}')
        # variables
        installTechnology  = getattr(model, f'install{tech}')
        capacityTechnology = getattr(model, f'capacity{tech}')

        return (maxCapacityTechnology * installTechnology[node, aliasNode, time]
                    >= capacityTechnology[node, aliasNode, time])


    #%% Contraint rules defined in current class - Operation
    @staticmethod
    def constraintTransportTechnologyFlowCapacityRule(model, tech, carrier, node, nodeAlias, time):
        """coupling flow carrier to capacity transport technology"""

        # variables
        capacityTechnology    = getattr(model, f'capacity{tech}')
        carrierFlowTechnology = getattr(model, f'carrierFlow{tech}')

        return (carrierFlowTechnology[carrier, node, nodeAlias, time]
                    <= capacityTechnology[node, nodeAlias, time])

    # Rules in current class
    @staticmethod
    def constraintTransportTechnologyLinearCapexValueRule(model, tech, node, aliasNode, time):
        """ definition of capex variable appearing in objective function"""

        # parameters
        distanceEucledian = getattr(model, f'distanceEucledian{tech}')
        costPerDistance   = getattr(model, f'costPerDistance{tech}')
        # variables
        capacityTechnology = getattr(model, f'capacity{tech}')

        return (model.capexTransportTechnology[tech, node, aliasNode, time] == 0.5 *
                capacityTechnology[node, aliasNode, time] *
                distanceEucledian[node, aliasNode, time] *
                costPerDistance[node, aliasNode, time])

    # # operational constraints
    # @staticmethod
    # def constraintMinLoadTransportTechnologies1Rule(model, carrier, tech, node, aliasNode, time):
    #     """min flow through transport technology between two nodes.
    #     \n\t Dimensions: setTransportCarriers, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps"""
    #     return (model.minTransportTechLoad[tech] * model.installTransportTechnologies[tech, node, aliasNode, time] * model.minCapacityTransport[tech]
    #             <= model.carrierFlowAux[carrier, tech, node, aliasNode, time])

    # def constraintMinLoadTransportTechnologies2Rule(model, carrier, tech, node, aliasNode, time):
    #     """min amount of carrier flow thorugh transport technology between two nodes.
    #     \n\t Dimensions: setCarrier, setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps"""
    #     return (model.carrierFlow[carrier, tech, node, aliasNode, time] - model.maxCapacityTransport[tech] * (1 - model.installTransportTechnologies[tech, node, aliasNode, time])
    #             <= model.carrierFlowAux[carrier, tech, node, aliasNode, time])
    
    # def constraintMaxLoadTransportTechnologies1Rule(model, carrier, tech, node, aliasNode, time):
    #     """max amount of carrier flow through transport technology between two nodes.
    #      \n\t Dimensions: setCarrier, setTransportTechnologiesnologies, setNodes, setAliasNodes, setTimeSteps"""
    #     return (model.capacityTransportTechnologies[tech, node, aliasNode, time]
    #             >= model.carrierFlowAux[carrier, tech, node, aliasNode, time])                                           # ub
    
    # def constraintMaxLoadTransportTechnologies2Rule(model, carrier, tech, node, aliasNode, time):
    #     """max amount of carrier flow through transport technology between two nodes.
    #     \n\t Dimensions: setCarrier, setTransportTechnologiesnologies, setNodes, setAliasNodes, setTimeSteps"""
    #     return (model.carrierFlow[carrier, tech, node, aliasNode, time]
    #             >= model.carrierFlowAux[carrier, tech, node, aliasNode, time])

    # %% Constraint rules defined in current class - Capital Expenditures (Capex)
