"""===========================================================================================================================================================================
Title:          ENERGY-CARBON OPTIMIZATION PLATFORM
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining the parameters, variables and constraints that hold for all transport technologies.
                The class takes the abstract optimization model as an input, and returns the parameters, variables and
                constraints that hold for the transport technologies.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
import numpy as np
from model.objects.technology.technology import Technology

class TransportTechnology(Technology):
    # empty list of elements
    listOfElements = []
    
    def __init__(self, object, tech):
        """init generic technology object
        :param object: object of the abstract optimization model"""

        logging.info('initialize object of a transport technology')
        super().__init__(object, tech)

        # set attributes of transport technology
        self.distance = object.pyoDict["distance"][tech]
        self.costPerDistance = object.pyoDict["costPerDistance"][tech]
        self.minFlow = object.pyoDict["minFlow"][tech]
        self.maxFlow = object.pyoDict["maxFlow"][tech]
        self.lossFlow = object.pyoDict["lossFlow"][tech]
        
        # add TransportTechnology to list
        TransportTechnology.addElement(self)

    ### --- classmethods to define sets, parameters, variables, and constraints, that correspond to TransportTechnology --- ###
    @classmethod
    def defineSets(cls):
        """ defines the pe.Sets of the class <TransportTechnology> """
        model = cls.getConcreteModel()
        pyoDict = cls.getPyoDict()

        # technologies and respective transport carriers
        model.setTransportCarriersTech = pe.Set(
            initialize = [(tech,pyoDict["setTransportCarriers"][tech]) for tech in pyoDict["setTransportCarriers"]],
            doc='set of techs and their respective transport carriers.')

        # transport carriers of technology
        model.setTransportCarriers = pe.Set(
            model.setTransportTechnologies,
            initialize = lambda _,tech: pyoDict["setTransportCarriers"][tech],
            doc="set of carriers that are transported in a specific transport technology")

    @classmethod
    def defineParams(cls):
        """ defines the pe.Params of the class <TransportTechnology> """
        model = cls.getConcreteModel()
        # distance between nodes
        model.distance = pe.Param(
            model.setTransportTechnologies,
            model.setEdges,
            initialize = cls.getAttributeOfAllElements("distance"),
            doc = 'distance between two nodes for transport technologies.\n\t Dimensions: setTransportTechnologies, setEdges')
        # cost per distance
        model.costPerDistance = pe.Param(
            model.setTransportTechnologies,
            model.setEdges,
            model.setTimeSteps,
            initialize = cls.getAttributeOfAllElements("costPerDistance"),
            doc = 'capex per unit distance for transport technologies. \n\t Dimensions: setTransportTechnologies, setEdges, setTimeSteps')
        # minimum flow relative to capacity
        model.minFlow = pe.Param(
            model.setTransportTechnologies,
            initialize = cls.getAttributeOfAllElements("minFlow"),
            doc = 'minimum flow through the transport technologies relative to installed capacity.\n\t Dimensions: setTransportTechnologies')
        # maximum flow relative to capacity
        model.maxFlow = pe.Param(
            model.setTransportTechnologies,
            initialize = cls.getAttributeOfAllElements("maxFlow"),
            doc = 'maximum flow through the transport technologies relative to installed capacity.\n\t Dimensions: setTransportTechnologies')
        # carrier losses
        model.lossFlow = pe.Param(
            model.setTransportTechnologies,
            initialize = cls.getAttributeOfAllElements("lossFlow"),
            doc = 'carrier losses due to transport with transport technologies.\n\t Dimensions: setTransportTechnologies')

    @classmethod
    def defineVars(cls):
        """ defines the pe.Vars of the class <TransportTechnology> """
        model = cls.getConcreteModel()
        # flow of carrier on edge
        model.carrierFlow = pe.Var(
            model.setTransportCarriersTech,
            model.setEdges,
            model.setTimeSteps,
            domain = pe.NonNegativeReals,
            doc = 'carrier flow through transport technology on edge i and time t.  \n\t Dimensions: setTransportCarriersTech, setEdges, setTimeSteps. \n\t Domain: NonNegativeReals'
        )
        # loss of carrier on edge
        model.carrierLoss = pe.Var(
            model.setTransportCarriersTech,
            model.setEdges,
            model.setTimeSteps,
            domain = pe.NonNegativeReals,
            doc = 'carrier flow through transport technology on edge i and time t.  \n\t Dimensions: setTransportCarriersTech, setEdges, setTimeSteps. \n\t Domain: NonNegativeReals'
        )
        # auxiliary variable of available capacity
        model.capacityAux = pe.Var(
            model.setTransportTechnologies,
            model.setEdges,
            model.setTimeSteps,
            domain = pe.NonNegativeReals,
            doc = 'auxiliary variable of the available capacity to model min and max possible flow through transport technologies. \n\t Dimensions: setTransportTechnologies,setEdges, setTimeSteps. \n\t Domain: NonNegativeReals'
        )
        # binary select variable
        model.select = pe.Var(
            model.setTransportTechnologies,
            model.setEdges,
            model.setTimeSteps,
            domain = pe.Binary,
            doc = 'binary variable to model the scheduling of transport technologies.  \n\t Dimensions: setTransportTechnologies, setEdges, setTimeSteps. \n\t Domain: Binary'
        )
        
    @classmethod
    def defineConstraints(cls):
        """ defines the pe.Constraints of the class <TransportTechnology> """
        model = cls.getConcreteModel()
        # min flow
        model.constraintTransportTechnologyMinFlow = pe.Constraint(
            model.setTransportCarriersTech,
            model.setEdges,
            model.setTimeSteps,
            rule = constraintTransportTechnologyMinFlowRule,
            doc = 'min possible carrier flow through transport technology. \n\t Dimensions: setTransportCarriersTech, setEdges, setTimeSteps'
        )
        # max flow
        model.constraintTransportTechnologyMaxFlow = pe.Constraint(
            model.setTransportCarriersTech,
            model.setEdges,
            model.setTimeSteps,
            rule = constraintTransportTechnologyMaxFlowRule,
            doc = 'max possible carrier flow through transport technology. \n\t Dimensions: setTransportCarriersTech, setEdges, setTimeSteps'
        )
        # Selection 1
        model.constraintTransportTechnologySelection1 = pe.Constraint(
            model.setTransportTechnologies,
            model.setEdges,
            model.setTimeSteps,
            rule = constraintTransportTechnologySelection1Rule,
            doc = 'select if transport technology is used or not Part 1. \n\t Dimensions: setTransportTechnologies, setEdges, setTimeSteps'
        )     
        # Selection 2
        model.constraintTransportTechnologySelection2 = pe.Constraint(
            model.setTransportTechnologies,
            model.setEdges,
            model.setTimeSteps,
            rule = constraintTransportTechnologySelection2Rule,
            doc = 'select if transport technology is used or not Part 2. \n\t Dimensions: setTransportTechnologies, setEdges, setTimeSteps'
        )        
        # AuxLBFlow 
        model.constraintTransportTechnologyAuxLBFlow = pe.Constraint(
            model.setTransportTechnologies,
            model.setEdges,
            model.setTimeSteps,
            rule = constraintTransportTechnologyAuxLBFlowRule,
            doc = 'LB for auxiliary variable capacityAux. \n\t Dimensions: setTransportTechnologies, setEdges, setTimeSteps'
        )     
        # AuxUBFlow 
        model.constraintTransportTechnologyAuxUBFlow = pe.Constraint(
            model.setTransportTechnologies,
            model.setEdges,
            model.setTimeSteps,
            rule = constraintTransportTechnologyAuxUBFlowRule,
            doc = 'UB for auxiliary variable capacityAux. \n\t Dimensions: setTransportTechnologies, setEdges, setTimeSteps'
        )   
        # Carrier Flow Losses 
        model.constraintTransportTechnologyLossesFlow = pe.Constraint(
            model.setTransportCarriersTech,
            model.setEdges,
            model.setTimeSteps,
            rule = constraintTransportTechnologyLossesFlowRule,
            doc = 'Carrier loss due to transport with through transport technology. \n\t Dimensions: setTransportCarriersTech, setEdges, setTimeSteps'
        ) 
        # Linear Capex
        model.constraintTransportTechnologyLinearCapex = pe.Constraint(
            model.setTransportTechnologies,
            model.setEdges,
            model.setTimeSteps,
            rule = constraintTransportTechnologyLinearCapexRule,
            doc = 'Capital expenditures for installing transport technology. \n\t Dimensions: setTransportTechnologies, setEdges, setTimeSteps'
        ) 

#%% Contraint rules defined in current class - Operation
def constraintTransportTechnologyMinFlowRule(model, tech, carrier, edge, time):
    """min carrier flow through transport technology"""
    if model.minFlow[tech] != 0:
        return (model.carrierFlow[tech,carrier, edge, time]
                >= model.minFlow[tech] * model.capacityAux[tech,edge, time])
    else:
        return pe.Constraint.Skip

def constraintTransportTechnologyMaxFlowRule(model, tech, carrier, edge, time):
    """max flow carrier through  transport technology"""
    if model.maxFlow[tech] != np.inf:
        return (model.carrierFlow[tech,carrier, edge, time]
                <= model.maxFlow[tech] * model.capacityAux[tech,edge, time])
    else:
        return pe.Constraint.Skip

def constraintTransportTechnologySelection1Rule(model, tech, edge, time):
    """select if transport technology is used or not"""
    if model.minCapacity[tech] != 0:
        return (model.capacityAux[tech,edge, time]
                >= model.minCapacity[tech] * model.select[tech,edge, time])
    else:
        return pe.Constraint.Skip

def constraintTransportTechnologySelection2Rule(model, tech, edge, time):
    """select if transport technology is used or not"""
    if model.maxCapacity[tech] != np.inf:
        return (model.capacityAux[tech,edge, time]
                <= model.maxCapacity[tech] * model.select[tech,edge, time])
    else:
        return pe.Constraint.Skip

def constraintTransportTechnologyAuxLBFlowRule(model, tech, edge, time):
    """coupling capacity and auxiliary capacity variable of transport technology"""
    return (model.capacityTechnology[tech,edge, time]
            - model.maxCapacity[tech] * (1 - model.select[tech,edge, time])
            <= model.capacityAux[tech,edge, time])

def constraintTransportTechnologyAuxUBFlowRule(model, tech, edge, time):
    """coupling capacity and auxiliary capacity variable of transport technology"""
    return (model.capacityAux[tech,edge, time]
            <= model.capacityTechnology[tech,edge, time])

def constraintTransportTechnologyLossesFlowRule(model, tech, carrier, edge, time):
    """compute the flow losses for a carrier through a transport technology"""
    return(model.carrierLoss[tech,carrier, edge, time]
            == model.distance[tech,edge] * model.lossFlow[tech] * model.carrierFlow[tech,carrier, edge, time])

def constraintTransportTechnologyLinearCapexRule(model, tech, edge, time):
    """ definition of the capital expenditures for the transport technology"""
    return (model.capexTechnology[tech,edge, time] == 0.5 *
            model.capacityTechnology[tech,edge, time] *
            model.distance[tech,edge] *
            model.costPerDistance[tech,edge, time])

