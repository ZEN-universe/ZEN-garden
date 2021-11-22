"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      November-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class containing the available objective functions as attributes
==========================================================================================================================================================================="""
from model.model_instance.aux_functions import hassattr
import pyomo.environ as pe

class ObjectiveFunction:
    
    def __init__(self, analysis, model):
        
        objFunc  = analysis['objective']
        objSense = analysis['sense']
        objRule  = 'objective' + objFunc + 'Rule'
        peObj    = pe.Objective(rule =  getattr(self, objRule),
                                sense = getattr(pe,   objSense))
        setattr(model, objFunc, peObj)

    def objectiveTotalCostRule(self,):
        """
        :return:
        """
        model = self.model
    
        # carrier
        carrierCost = sum(sum(sum(model.importCarrier[carrier, node, time] * model.price[carrier, node, time]
                                for time in model.setTimeSteps)
                            for node in model.setNodes)
                        for carrier in model.setCarriersIn)
    
        # production and storage techs
        installCost = 0
        for techType in ['Production', 'Storage']:
            if hassattr(model, f'set{techType}Technologies'):
                installCost += sum(sum(sum(model.installProductionTech[tech, node, time]
                                           for time in model.setTimeSteps)
                                        for node in model.setNodes)
                                    for tech in getattr(model, f'set{techType}Technologies'))
    
        # transport techs
        if hassattr(model, 'setTransport'):
            installCost += sum(sum(sum(sum(model.installProductionTech[tech, node, aliasNode, time]
                                            for time in model.setTimeSteps)
                                        for node in model.setNodes)
                                    for aliasNode in model.setAliasNodes)
                                for tech in model.setTransport)
    
        return(carrierCost + installCost)        
    
    def objectiveCarbonEmissionsRule(self):
        """
        :return:
        """
        model = self.model        
        # TODO implement objective functions for emissions
        pass
