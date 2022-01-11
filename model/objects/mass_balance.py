"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      November-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class containing the mass balance and its attributes.
==========================================================================================================================================================================="""
from model.objects.element import Element
import pyomo.environ as pe

class MassBalance(Element):

    def __init__(self, object):
        """initialization of the mass balance
        :param object: object of the abstract optimization model """

        super().__init__(object)

        #%% Contraints
        constraint = {'NodalMassBalance':    'nodal mass balance for each time step. \
                                             \n\t Dimensions: setCarriers, setNodes, setTimeSteps'
            }
        self.addConstr(constraint)  


    #%% Constraint rules defined in current class
    @staticmethod
    def constraintNodalMassBalanceRule(model, carrier, node, time):
        """" 
        nodal mass balance for each time step
        """

        # carrier input and output conversion technologies
        carrierConversionIn, carrierConversionOut = 0, 0
        if hasattr(model, 'setConversionTechnologies'):
            for tech in model.setConversionTechnologies:
                inputTechnology  = getattr(model, f'input{tech}')
                outputTechnology = getattr(model, f'output{tech}')
                if carrier in getattr(model, f'setInputCarriers{tech}'):
                    carrierConversionIn += inputTechnology[carrier, node, time]
                if carrier in getattr(model, f'setOutputCarriers{tech}'):
                    carrierConversionOut += outputTechnology[carrier, node, time]

        # carrier flow transport technologies
        carrierFlowIn, carrierFlowOut = 0, 0
        if hasattr(model, 'setTransportTechnologies'):
            for tech in model.setTransportTechnologies:
                carrierFlow = getattr(model, f'carrierFlow{tech}')
                carrierLoss = getattr(model, f'carrierLoss{tech}')
                if carrier in getattr(model, f'setTransportCarriers{tech}'):
                    carrierFlowIn += sum(carrierFlow[carrier, aliasNode, node, time]
                                         - carrierLoss[carrier, aliasNode, node, time] for aliasNode in model.setAliasNodes)
                    carrierFlowOut += sum(carrierFlow[carrier, node, aliasNode, time] for aliasNode in model.setAliasNodes)

        # carrier import, demand and export
        carrierImport, carrierExport, carrierDemand = 0, 0, 0
        if carrier in model.setInputCarriers:
            carrierImport = model.importCarrier[carrier, node, time]
        if carrier in model.setOutputCarriers:
            carrierDemand = model.demandCarrier[carrier, node, time]
            carrierExport = model.exportCarrier[carrier, node, time]

        # TODO implement storage

        return (carrierConversionOut - carrierConversionIn + carrierFlowIn - carrierFlowOut + carrierImport - carrierExport - carrierDemand == 0)