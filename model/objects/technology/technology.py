"""===========================================================================================================================================================================
Title:          ENERGY-CARBON OPTIMIZATION PLATFORM
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining the parameters, variables and constraints that hold for all technologies.
                The class takes the abstract optimization model as an input, and returns the parameters, variables and
                constraints that hold for all technologies.
==========================================================================================================================================================================="""

import logging
import sys
import pyomo.environ as pe
import numpy as np
from model.objects.element import Element

class Technology(Element):
    # empty list of elements
    listOfElements = []

    def __init__(self, object, technology):
        """init generic technology object
        :param object: object of the abstract optimization model
        :param technology: technology that is added to the model"""

        logging.info('initialize object of a generic technology')
        super().__init__(object,technology)
        # add Technology to list
        Technology.addElement(self)
    
    def storeInputData(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """   
        # get system information
        system = Element.getSystem()   
        paths = Element.getPaths()   
        indexNames = Element.getAnalysis()['dataInputs']
        technologyTypes = Element.getAnalysis()['subsets']["setTechnologies"]
        # set attributes of technology
        # parameters
        for technologyType in technologyTypes:
            if self.name in system[technologyType]:
                _inputPath = paths[technologyType][self.name]["folder"]
                self.minCapacity = Element.extractAttributeData(_inputPath,"minCapacity")
                self.maxCapacity = Element.extractAttributeData(_inputPath,"maxCapacity")
                self.lifetime = Element.extractAttributeData(_inputPath,"lifetime")
                if technologyType == "setConversionTechnologies":
                    self.referenceCarrier = Element.extractAttributeData(_inputPath,"referenceCarrier")
                    self.availability = Element.extractInputData(_inputPath,"availability",[indexNames["nameNodes"],indexNames["nameTimeSteps"]])
                elif technologyType == "setTransportTechnologies":
                    self.referenceCarrier = Element.extractAttributeData(_inputPath,"referenceCarrier")
                    # transport data
                    # self.availability = Element.extractInputData(_inputPath,"availability",[indexNames["nameNodes"],indexNames["nameTimeSteps"]])


    ### --- classmethods to define sets, parameters, variables, and constraints, that correspond to Technology --- ###
    @classmethod
    def defineSets(cls):
        """ defines the pe.Sets of the class <Technology> """
        # define the pe.Sets of the class <Technology>
        model = cls.getConcreteModel()
        
        # conversion technologies
        model.setConversionTechnologies = pe.Set(
            initialize=Element.getAttributeOfElement("grid","setConversionTechnologies"), 
            doc='Set of conversion technologies. Subset: setTechnologies')
        # transport technologies
        model.setTransportTechnologies = pe.Set(
            initialize=Element.getAttributeOfElement("grid","setTransportTechnologies"), 
            doc='Set of transport technologies. Subset: setTechnologies')
        # combined technology and location set
        model.setTechnologyLocation = pe.Set(
            initialize = technologyLocationRule,
            doc = "Combined set of technologies and locations. Conversion technologies are paired with nodes, transport technologies are paired with edges"
        )

        # add pe.Sets of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.defineSets()

    @classmethod
    def defineParams(cls):
        """ defines the pe.Params of the class <Technology> """
        ### define pe.Param of the class <Technology>
        model = cls.getConcreteModel()
        
        # minimum capacity
        model.minCapacity = pe.Param(
            model.setTechnologies,
            initialize = cls.getAttributeOfAllElements("minCapacity"),
            doc = 'Parameter which specifies the minimum technology size that can be installed.\n\t Dimensions: setTechnologies')
        # maximum capacity
        model.maxCapacity = pe.Param(
            model.setTechnologies,
            initialize = cls.getAttributeOfAllElements("maxCapacity"),
            doc = 'Parameter which specifies the maximum technology size that can be installed.\n\t Dimensions: setTechnologies')
        # lifetime
        model.lifetimeTechnology = pe.Param(
            model.setTechnologies,
            initialize = cls.getAttributeOfAllElements("lifetime"),
            doc = 'Parameter which specifies the lifetime of technology.\n\t Dimensions: setTechnologies')
        # availability of  technologies
        model.availabilityTechnology = pe.Param(
            model.setTechnologyLocation,
            model.setTimeSteps,
            initialize = cls.getAttributeOfAllElements("availability"),
            doc = 'Parameter which specifies the availability of technologies.\n\t Dimensions: setTechnologyLocation, setTimeSteps')

        # add pe.Param of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.defineParams()

    @classmethod
    def defineVars(cls):
        """ defines the pe.Vars of the class <Technology> """
        model = cls.getConcreteModel()
        # define pe.Vars of the class <Technology>
        # install technology
        model.installTechnology = pe.Var(
            model.setTechnologyLocation,
            model.setTimeSteps,
            domain = pe.Binary,
            doc = 'installment of a technology on edge i and time t.  \n\t Dimensions: setTechnologyLocation, setTimeSteps.\n\t Domain: Binary')
        # capacity technology
        model.capacityTechnology = pe.Var(
            model.setTechnologyLocation,
            model.setTimeSteps,
            domain = pe.NonNegativeReals,
            doc = 'size of installed technology on edge i and time t.  \n\t Dimensions: setTechnologyLocation, setTimeSteps.\n\t Domain: NonNegativeReals')
        # builtCapacity technology
        model.builtCapacityTechnology = pe.Var(
            model.setTechnologyLocation,
            model.setTimeSteps,
            domain = pe.NonNegativeReals,
            doc = 'size of built technology on edge i and time t.  \n\t Dimensions: setTechnologyLocation, setTimeSteps.\n\t Domain: NonNegativeReals')
        # capex technology
        model.capexTechnology = pe.Var(
            model.setTechnologyLocation,
            model.setTimeSteps,
            domain = pe.NonNegativeReals,
            doc = 'capex for installing technology on edge i and time t.  \n\t Dimensions: setTechnologyLocation, setTimeSteps.\n\t Domain: NonNegativeReals')
        # total capex technology
        model.capexTotalTechnology = pe.Var(
            domain = pe.NonNegativeReals,
            doc = 'total capex for installing all technologies on all edges at all times.  \n\t Domain: NonNegativeReals')

        # add pe.Vars of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.defineVars()

    @classmethod
    def defineConstraints(cls):
        """ defines the pe.Constraints of the class <Technology> """
        model = cls.getConcreteModel()
        # define pe.Constraints of the class <Technology>
        #  technology availability
        model.constraintTechnologyAvailability = pe.Constraint(
            model.setTechnologyLocation,
            model.setTimeSteps,
            rule = constraintTechnologyAvailabilityRule,
            doc = 'limited availability of  technology depending on node and time. \n\t Dimensions: setTechnologyLocation, setTimeSteps'
        )
        # minimum capacity
        model.constraintTechnologyMinCapacity = pe.Constraint(
            model.setTechnologyLocation,
            model.setTimeSteps,
            rule = constraintTechnologyMinCapacityRule,
            doc = 'min capacity of  technology that can be installed. \n\t Dimensions: setTechnologyLocation, setTimeSteps'
        )
        # maximum capacity
        model.constraintTechnologyMaxCapacity = pe.Constraint(
            model.setTechnologyLocation,
            model.setTimeSteps,
            rule = constraintTechnologyMaxCapacityRule,
            doc = 'max capacity of  technology that can be installed. \n\t Dimensions: setTechnologyLocation, setTimeSteps'
        )
        
        # lifetime
        model.constraintTechnologyLifetime = pe.Constraint(
            model.setTechnologyLocation,
            model.setTimeSteps,
            rule = constraintTechnologyLifetimeRule,
            doc = 'max capacity of  technology that can be installed. \n\t Dimensions: setTechnologyLocation, setTimeSteps'
        )
        # total capex of all technologies
        model.constraintCapexTotalTechnology = pe.Constraint(
            rule = constraintCapexTotalTechnologyRule,
            doc = 'total capex of all technology that can be installed.'
        )
        # add pe.Constraints of the child classes
        for subclass in cls.getAllSubclasses():
            subclass.defineConstraints()

# function to combine the technologies and locations
def technologyLocationRule(model):
    """ creates list for setTechnologyLocation, where ConversionTechnologies are paired with the nodes and TransportTechnologies are paired with edges
    :return technologyLocationList: list of 2-tuple with (technology, location)"""
    technologyLocationList = [(technology,location) for technology in model.setConversionTechnologies for location in model.setNodes]
    technologyLocationList.extend([(technology,location) for technology in model.setTransportTechnologies for location in model.setEdges])
    return technologyLocationList


### --- staticmethods with constraint rules --- ###
#%% Constraint rules pre-defined in Technology class
def constraintTechnologyAvailabilityRule(model, tech, location, time):
    """limited availability of  technology"""
    if model.availabilityTechnology[tech, location, time] != np.inf:
        return (model.availabilityTechnology[tech, location, time] >= model.installTechnology[tech, location, time])
    else:
        return pe.Constraint.Skip

def constraintTechnologyMinCapacityRule(model, tech, location, time):
    """ min capacity expansion of  technology."""
    if model.minCapacity[tech] != 0:
        return (model.minCapacity[tech] * model.installTechnology[tech, location, time]
                <= model.capacityTechnology[tech, location, time])
    else:
        return pe.Constraint.Skip

def constraintTechnologyMaxCapacityRule(model, tech, location, time):
    """max capacity expansion of  technology"""
    if model.maxCapacity[tech] != np.inf:
        return (model.maxCapacity[tech] * model.installTechnology[tech, location, time]
                >= model.capacityTechnology[tech, location, time])
    else:
        return pe.Constraint.Skip

def constraintTechnologyLifetimeRule(model, tech, location, time):
    """limited lifetime of the technologies"""
    # time range
    t_start = max(0, time - model.lifetimeTechnology[tech] + 1)
    t_end = time + 1

    return (model.capacityTechnology[tech, location, time]
            == sum(model.builtCapacityTechnology[tech,location, t] for t in range(t_start, t_end)))

def constraintCapexTotalTechnologyRule(model):
    """ sums over all technologies to calculate total capex """
    return(model.capexTotalTechnology == 
        sum(
            sum(
                model.capexTechnology[tech, loc,time]
                for tech,loc in model.setTechnologyLocation
            )
            for time in model.setTimeSteps
        )
    )

### TODO fix from here ###
def constraintTechnologyMinCapacityExpansionRule(model, tech, location, time):
    """min capacity expansion of conversion technology"""

    # parameters
    minCapacityExpansion = getattr(model, f'minCapacityExpansion{tech}')
    # variables
    expandTechnology = getattr(model, f'expand{tech}')
    

    return (expandTechnology[location, t] * minCapacityExpansion #TODO what is t, fix!
            >= model.builtCapacityTechnologies[tech, location, time])

def constraintTechnologyMaxCapacityExpansionRule(model, tech, location, time):
    """max capacity expansion of conversion technology"""

    # parameters
    maxCapacityExpansion = getattr(model, f'maxCapacityExpansion{tech}')
    # variables
    expandTechnology = getattr(model, f'expand{tech}')

    return (expandTechnology[location, t] * maxCapacityExpansion #TODO what is t, fix!
            <= model.builtCapacityTechnologies[tech, location, time])

def constraintConversionTechnologyLimitedCapacityExpansionRule(model, tech, location, time):
    """technology capacity can only be expanded once during its lifetime"""

    # parameters
    lifetime = getattr(model, f'lifetime{tech}')
    # variables
    installTechnology = getattr(model, f'install{tech}')
    expandTechnology  = getattr(model, f'expandit{tech}')

    # time range
    t_start = max(1, t - lifetime + 1)
    t_end = time + 1

    return (sum(expandTechnology[location, t] for t in range(t_start, t_end)) <= installTechnology[location, t])

### TODO fix until here