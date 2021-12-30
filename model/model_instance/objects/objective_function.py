"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      November-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
              Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:   Class containing the available objective function and its attributes.
==========================================================================================================================================================================="""

from model.model_instance.objects.element import Element
import pyomo.environ as pe

class ObjectiveFunction(Element):

    def __init__(self, object):
        """initialization of the objective function
        :param object: object of the abstract optimization model"""

        super().__init__(object)

        objFunc  = self.analysis['objective']
        objSense = self.analysis['sense']
        objRule  = 'objective' + objFunc + 'Rule'
        peObj    = pe.Objective(rule =  getattr(self, objRule),
                                sense = getattr(pe,   objSense))
        setattr(self.model, objFunc, peObj)

    # RULES
    @staticmethod
    def objectiveBasicTotalCostRule(model):
        " basic cost rule with PWA capex and linear transport cost"

        print("*** Has attribute valueCapex", hasattr(model, 'valueCapex'))

        # Production cost
        installCost = 0
        for techType in ['Production']:
            if hasattr(model, f'set{techType}Technologies'):
                installCost += sum(sum(sum(
                    model.capexTechnologyValue[tech, node, time]
                    for time in model.setTimeSteps)
                        for node in model.setNodes)
                            for tech in getattr(model, f'set{techType}Technologies'))
                
        # Transport cost
        for techType in ['Transport']:
            if hasattr(model, f'set{techType}Technologies'):

                installCost += sum(sum(sum(
                    0.5*
                    model.capacityTransportTechnologies[tech, node, nodealias, time]*
                    model.distanceEucledian[tech, node, nodealias, time]*
                    model.costPerDistance[tech, node, nodealias, time]
                    for time in model.setTimeSteps)
                        for nodealias in model.setNodes)
                            for node in model.setAliasNodes
                                for tech in getattr(model, f'set{techType}Technologies'))
                
        return installCost
    
    @staticmethod
    def objectiveTotalCostRule(model):
        """objective function to minimize the total cost"""
    
        # CARRIERS
        carrierImport = sum(sum(sum(model.importCarrier[carrier, node, time] * model.importPriceCarrier[carrier, node, time]
                                for time in model.setTimeSteps)
                            for node in model.setNodes)
                        for carrier in model.setInputCarriers)

        carrierExport = sum(sum(sum(model.exportCarrier[carrier, node, time] * model.exportPriceCarrier[carrier, node, time]
                                for time in model.setTimeSteps)
                            for node in model.setNodes)
                        for carrier in model.setOutputCarriers)

        # PRODUCTION AND STORAGE TECHNOLOGIES
        installCost = 0
        for techType in ['Conversion', 'Storage']:
            if hasattr(model, f'set{techType}Technologies'):
                installCost += sum(sum(sum(model.installProductionTechnologies[tech, node, time]
                                           for time in model.setTimeSteps)
                                        for node in model.setNodes)
                                    for tech in getattr(model, f'set{techType}Technologies'))

        # TRANSPORT TECHNOLOGIES
        if hasattr(model, 'setTransport'):
            installCost += sum(sum(sum(sum(model.installTransportTechnologies[tech, node, aliasNode, time]
                                            for time in model.setTimeSteps)
                                        for node in model.setNodes)
                                    for aliasNode in model.setAliasNodes)
                                for tech in model.setTransport)
    
        return(carrierImport - carrierExport + installCost)

    @staticmethod
    def objectiveCarbonEmissionsRule(model):
        """objective function to minimize total emissions"""

        # TODO implement objective functions for emissions
        return pe.Constraint.Skip

    @staticmethod
    def objectiveRisk(model):
        """objective function to minimize total risk"""

        # TODO implement objective functions for risk
        return pe.Constraint.Skip
