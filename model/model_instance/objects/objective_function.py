"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      November-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class containing the available objective functions as attributes
==========================================================================================================================================================================="""
from model.model_instance.aux_functions import hassattr
from model.model_instance.objects.element import Element
import pyomo.environ as pe

class ObjectiveFunction(Element):

    def __init__(self, object):

        super().__init__(object)

        objFunc  = self.analysis['objective']
        objSense = self.analysis['sense']
        objRule  = 'objective' + objFunc + 'Rule'
        peObj    = pe.Objective(rule =  getattr(self, objRule),
                                sense = getattr(pe,   objSense))
        setattr(self.model, objFunc, peObj)

    @staticmethod
    def objectiveTotalCostRule(model):
        """
        objective function to minimize the total cost
        """
    
        # carrier
        # carrierCost = sum(sum(sum(model.importCarrier[carrier, node, time] * model.price[carrier, node, time]
        #                         for time in model.setTimeSteps)
        #                     for node in model.setNodes)
        #                 for carrier in model.setCarriersIn)
        #
        # # production and storage techs
        # installCost = 0
        # for techType in ['Production', 'Storage']:
        #     if hassattr(model, f'set{techType}Technologies'):
        #         installCost += sum(sum(sum(model.installProductionTech[tech, node, time]
        #                                    for time in model.setTimeSteps)
        #                                 for node in model.setNodes)
        #                             for tech in getattr(model, f'set{techType}Technologies'))
        #
        # # transport techs
        # if hassattr(model, 'setTransport'):
        #     installCost += sum(sum(sum(sum(model.installProductionTech[tech, node, aliasNode, time]
        #                                     for time in model.setTimeSteps)
        #                                 for node in model.setNodes)
        #                             for aliasNode in model.setAliasNodes)
        #                         for tech in model.setTransport)
    
        return(1)#carrierCost + installCost)

    @staticmethod
    def objectiveCarbonEmissionsRule(model):
        """
        :return:
        """
        # TODO implement objective functions for emissions
        pass
