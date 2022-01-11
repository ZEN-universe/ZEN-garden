"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      November-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
              Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:   Class containing the available objective function and its attributes.
==========================================================================================================================================================================="""

from model.objects.element import Element
import pyomo.environ as pe

class ObjectiveFunction(Element):

    def __init__(self, object):
        """initialization of the objective function
        :param object: object of the abstract optimization model"""

        super().__init__(object)

        self.addAuxiliaryConstraints()

        objFunc  = self.analysis['objective']
        objSense = self.analysis['sense']
        objRule  = 'objective' + objFunc + 'Rule'
        peObj    = pe.Objective(rule =  getattr(self, objRule),
                                sense = getattr(pe,   objSense))
        setattr(self.model, objFunc, peObj)


    def addAuxiliaryConstraints(self):
        """add auxiliary constraints for more direct formulation of the objective function"""

        vars = {
            f'capexTransportTechnology': f'Capex of transport technologies used in definition of objective function. \
                                         \n\t Dimensions: setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps.\
                                         \n\t Domain: NonNegativeReals',
            'capexConversionTechnology': f'Capex of conversion technologies used in definition of objective function. \
                                         \n\t Dimensions: setConversionTechnologies, setNodes, setTimeSteps.\
                                         \n\t Domain: NonNegativeReals'}
        self.addVars(vars)

        constr = {
            f'TransportTechnologyLinearCapexValue': f'Definition of Capex for all the transport technologies.\
                                                    \n\t Dimensions: setTransportTechnologies, setNodes, setAliasNodes, setTimeSteps.',
            f'ConversionTechnologyLinearCapexValue': f'Definition of Capex for all the conversion technologies.\
                                                    \n\t Dimensions: setConversionTechnologies, setNodes, setTimeSteps.'}
        self.addConstr(constr)

    #%% Constraint rules defined in current class - Auxiliary constraints
    @staticmethod
    def constraintConversionTechnologyLinearCapexValueRule(model, tech, node, time):
        """definition of capex variable appearing in objective function"""

        # variables
        capexConversionTechnology = getattr(model, f'capex{tech}')

        return(model.capexConversionTechnology[tech, node, time]
               == capexConversionTechnology[node, time])

    @staticmethod
    def constraintTransportTechnologyLinearCapexValueRule(model, tech, node, aliasNode, time):
        """ definition of capex variable appearing in objective function"""

        # variables
        capexTransportTechnology = getattr(model, f'capex{tech}')

        return(model.capexTransportTechnology[tech, node, aliasNode, time]
               == capexTransportTechnology[node, aliasNode, time])

    #%% Objective functions
    @staticmethod
    def objectiveBasicTotalCostRule(model):
        " basic cost rule with PWA capex and linear transport cost"

        installCost = 0

        # Capex conversion technologies
        installCost += sum(sum(sum(
            model.capexConversionTechnology[tech, node, time]
            for time in model.setTimeSteps)
                for node in model.setNodes)
                    for tech in getattr(model, f'setConversionTechnologies'))
                
        # Capex transport technologies
        installCost += sum(sum(sum(sum(
            model.capexTransportTechnology[tech, node, aliasNode, time]
            for time in model.setTimeSteps)
                for node in model.setAliasNodes)
                    for aliasNode in model.setNodes)
                        for tech in getattr(model, f'setTransportTechnologies'))
                
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

        # Capex Conversion and Storage technologies
        technologies = ['Conversion', 'Storage']
        for tech in technologies:
            if hasattr(model, f'set{tech}Technologies'):
                installCost += sum(sum(sum(get(model, f'capex{tech}Technology')[tech, node, time]
                                           for time in model.setTimeSteps)
                                       for node in model.setNodes)
                                   for tech in getattr(model, f'setConversionTechnologies'))


        # Capex transport technologies
        if hasattr(model, f'setTransportTechnologies'):
            installCost += sum(sum(sum(sum(model.capexTransportTechnology[tech, node, aliasNode, time]
                                           for time in model.setTimeSteps)
                                       for node in model.setAliasNodes)
                                   for aliasNode in model.setNodes)
                               for tech in getattr(model, f'setTransportTechnologies'))
    
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
