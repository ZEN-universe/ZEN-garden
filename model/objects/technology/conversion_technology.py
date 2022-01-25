"""===========================================================================================================================================================================
Title:          ENERGY-CARBON OPTIMIZATION PLATFORM
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class defining the parameters, variables and constraints of the conversion technologies.
                The class takes the abstract optimization model as an input, and adds parameters, variables and
                constraints of the conversion technologies.
==========================================================================================================================================================================="""

import logging
import pyomo.environ as pe
import pyomo.gdp as pgdp
import numpy as np
from model.objects.technology.technology import Technology
from model.objects.element import Element
from model.objects.energy_system import EnergySystem

class ConversionTechnology(Technology):
    # empty list of elements
    listOfElements = []

    def __init__(self, tech):
        """init generic technology object
        :param object: object of the abstract model"""

        logging.info('initialize object of a conversion technology')
        super().__init__(tech)
        # store input data
        self.storeInputData()
        # add ConversionTechnology to list
        ConversionTechnology.addElement(self)

    def storeInputData(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """   
        # get attributes from class <Technology>
        super().storeInputData()
        # get system information
        paths       = EnergySystem.getPaths()   
        indexNames  = EnergySystem.getAnalysis()['dataInputs']

        # set attributes of technology
        _inputPath              = paths["setConversionTechnologies"][self.name]["folder"]
        self.deltaCapacity      = self.dataInput.extractAttributeData(_inputPath,"deltaCapacity")
        self.minLoad            = self.dataInput.extractAttributeData(_inputPath,"minLoad")
        self.maxLoad            = self.dataInput.extractAttributeData(_inputPath,"maxLoad")
        self.availability       = self.dataInput.extractInputData(_inputPath,"availability",[indexNames["nameNodes"],indexNames["nameTimeSteps"]])
        # define input and output carrier
        self.inputCarrier       = self.dataInput.extractConversionCarriers(_inputPath)["inputCarrier"]
        self.outputCarrier      = self.dataInput.extractConversionCarriers(_inputPath)["outputCarrier"]
        # extract PWA parameters
        self.PWAParameter       = self.dataInput.extractPWAData(_inputPath,self)

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to ConversionTechnology --- ###
    @classmethod
    def constructSets(cls):
        """ constructs the pe.Sets of the class <ConversionTechnology> """
        model = EnergySystem.getConcreteModel()
        # get input carriers
        _inputCarriers      = cls.getAttributeOfAllElements("inputCarrier")
        _outputCarriers     = cls.getAttributeOfAllElements("outputCarrier")
        _referenceCarrier   = cls.getAttributeOfAllElements("referenceCarrier")
        _PWAParameter       = cls.getAttributeOfAllElements("PWAParameter")
        _dependentCarriers  = {}
        for tech in _inputCarriers:
            _dependentCarriers[tech] = _inputCarriers[tech]+_outputCarriers[tech]
            _dependentCarriers[tech].remove(_referenceCarrier[tech][0])
        # input carriers of technology
        model.setInputCarriers = pe.Set(
            model.setConversionTechnologies,
            initialize = _inputCarriers,
            doc = "set of carriers that are an input to a specific conversion technology.\n\t Dimensions: setConversionTechnologies"
        )
        # output carriers of technology
        model.setOutputCarriers = pe.Set(
            model.setConversionTechnologies,
            initialize = _outputCarriers,
            doc = "set of carriers that are an output to a specific conversion technology.\n\t Dimensions: setConversionTechnologies"
        )
        # dependent carriers of technology
        model.setDependentCarriers = pe.Set(
            model.setConversionTechnologies,
            initialize = _dependentCarriers,
            doc = "set of carriers that are an output to a specific conversion technology.\n\t Dimensions: setConversionTechnologies"
        )
        # technologies and respective input carriers 
        model.setInputCarriersTechs = pe.Set(
            initialize = [(tech,inputCarrier) for tech in _inputCarriers for inputCarrier in _inputCarriers[tech]],
            doc = "set of techs and their respective input carriers"
        )
        # technologies and respective output carriers
        model.setOutputCarriersTechs = pe.Set(
            initialize = [(tech,outputCarrier) for tech in _outputCarriers for outputCarrier in _outputCarriers[tech]],
            doc = "set of techs and their respective output carriers"
        )

        ### sets for the definition of the approximation ##
        # set of PWA/NL technologies in capex approximation
        model.setPWACapexTechs = pe.Set(
            initialize=[tech for tech in model.setConversionTechnologies if tech not in EnergySystem.getAnalysis()["nonlinearTechnologyApproximation"]["Capex"] or EnergySystem.getSolver()["model"] == "MILP"],
            doc = "Set of conversion technologies for which the capex is PWA modeled")
        model.setNLCapexTechs = pe.Set(
            initialize=model.setConversionTechnologies - model.setPWACapexTechs,
            doc = "Set of conversion technologies for which the capex is NL modeled")
        # set of PWA/NL technologies in converEfficiency approximation
        model.setPWAConverEfficiencyTechs = pe.Set(
            initialize=[tech for tech in model.setConversionTechnologies if tech not in EnergySystem.getAnalysis()["nonlinearTechnologyApproximation"]["ConverEfficiency"] or EnergySystem.getSolver()["model"] == "MILP"],
            doc = "Set of conversion technologies for which the ConverEfficiency is PWA modeled")
        model.setNLConverEfficiencyTechs = pe.Set(
            initialize=model.setConversionTechnologies - model.setPWAConverEfficiencyTechs,
            doc = "Set of conversion technologies for which the ConverEfficiency is NL modeled")
        # set of variable indices in capex approximation
        model.setPWACapex = pe.Set(
            initialize = [(tech,node,timeStep)  for tech in model.setPWACapexTechs if "capex" in _PWAParameter[(tech,"Capex")]["PWAVariables"]
                                                for node in model.setNodes
                                                for timeStep in model.setTimeSteps],
            doc = "Set of capex indices for which the capex is PWA modeled"
        )
        model.setLinearCapex = pe.Set(
            initialize = [(tech,node,timeStep)  for tech in model.setPWACapexTechs if "capex" not in _PWAParameter[(tech,"Capex")]["PWAVariables"]
                                                for node in model.setNodes
                                                for timeStep in model.setTimeSteps],
            doc = "Set of capex indices for which the capex is linearly modeled"
        )
        model.setNLCapex = pe.Set(
            initialize = [(tech,node,timeStep)  for tech in model.setNLCapexTechs
                                                for node in model.setNodes
                                                for timeStep in model.setTimeSteps],
            doc = "Set of capex indices for which the capex is NL modeled"
        )
        # set of variable indices in ConverEfficiency approximation
        model.setPWAConverEfficiency = pe.Set(
            initialize = [(tech,dependentCarrier,node,timeStep) for tech in model.setPWAConverEfficiencyTechs 
                                                            for dependentCarrier in model.setDependentCarriers[tech] if dependentCarrier in _PWAParameter[(tech,"ConverEfficiency")]["PWAVariables"]
                                                            for node in model.setNodes
                                                            for timeStep in model.setTimeSteps],
            doc = "Set of ConverEfficiency indices for which the ConverEfficiency is PWA modeled"
        )
        model.setLinearConverEfficiency = pe.Set(
            initialize = [(tech,dependentCarrier,node,timeStep) for tech in model.setPWAConverEfficiencyTechs 
                                                            for dependentCarrier in model.setDependentCarriers[tech] if dependentCarrier not in _PWAParameter[(tech,"ConverEfficiency")]["PWAVariables"]
                                                            for node in model.setNodes
                                                            for timeStep in model.setTimeSteps],
            doc = "Set of ConverEfficiency indices for which the ConverEfficiency is linearly modeled"
        )
        model.setNLConverEfficiency = pe.Set(
            initialize = [(tech,dependentCarrier,node,timeStep) for tech in model.setNLConverEfficiencyTechs 
                                                            for dependentCarrier in model.setDependentCarriers[tech] 
                                                            for node in model.setNodes
                                                            for timeStep in model.setTimeSteps],
            doc = "Set of ConverEfficiency indices for which the ConverEfficiency is NL modeled"
        )
        
    @classmethod
    def constructParams(cls):
        """ constructs the pe.Params of the class <ConversionTechnology> """
        model = EnergySystem.getConcreteModel()
        _PWAParameter       = cls.getAttributeOfAllElements("PWAParameter")
        # slope of linearly modeled capex
        model.slopeLinearApproximationCapex = pe.Param(
            model.setLinearCapex,
            initialize = lambda _,tech,*__: _PWAParameter[(tech,"Capex")]["capex"]
        )
        # slope of linearly modeled conversion efficiencies
        model.slopeLinearApproximationConverEfficiency = pe.Param(
            model.setLinearConverEfficiency,
            initialize = lambda _,tech,carrier,*__: _PWAParameter[(tech,"ConverEfficiency")][carrier]
        )

    @classmethod
    def constructVars(cls):
        """ constructs the pe.Vars of the class <ConversionTechnology> """
        model = EnergySystem.getConcreteModel()
        
        ## Flow variables
        # input flow of carrier into technology
        model.inputFlow = pe.Var(
            model.setInputCarriersTechs,
            model.setNodes,
            model.setTimeSteps,
            domain = pe.NonNegativeReals,
            bounds = lambda _, tech, carrier, *__: cls.getAttributeOfAllElements("PWAParameter")[tech,"ConverEfficiency"]["bounds"][carrier],
            doc = 'Carrier input of conversion technologies. Dimensions: setInputCarriersTechs, setNodes, setTimeSteps. Domain: NonNegativeReals' )
        # output flow of carrier into technology
        model.outputFlow = pe.Var(
            model.setOutputCarriersTechs,
            model.setNodes,
            model.setTimeSteps,
            domain = pe.NonNegativeReals,
            bounds = lambda _, tech, carrier, *__: cls.getAttributeOfAllElements("PWAParameter")[tech,"ConverEfficiency"]["bounds"][carrier],
            doc = 'Carrier output of conversion technologies. Dimensions: setOutputCarriersTechs, setNodes, setTimeSteps. Domain: NonNegativeReals')

        ## PWA Variables - Capex
        # PWA capacity
        model.capacityApproximation = pe.Var(
            model.setPWACapex | model.setLinearCapex,
            domain = pe.NonNegativeReals,
            doc = 'PWA variable for size of installed technology on edge i and time t. Dimensions: setPWACapex | setLinearCapex. Domain: NonNegativeReals')
        # PWA capex technology
        model.capexApproximation = pe.Var(
            model.setPWACapex | model.setLinearCapex,
            domain = pe.NonNegativeReals,
            doc = 'PWA variable for capex for installing technology on edge i and time t. Dimensions:  setPWACapex | setLinearCapex. Domain: NonNegativeReals')

        ## PWA Variables - Conversion Efficiency
        # PWA reference flow of carrier into technology
        model.referenceFlowApproximation = pe.Var(
            model.setPWAConverEfficiency | model.setLinearConverEfficiency,
            domain = pe.NonNegativeReals,
            bounds = lambda model, tech, *__: cls.getAttributeOfAllElements("PWAParameter")[tech,"ConverEfficiency"]["bounds"][model.setReferenceCarriers[tech][1]],
            doc = 'PWA of flow of reference carrier of conversion technologies. Dimensions: setPWAConverEfficiency | setLinearConverEfficiency. Domain: NonNegativeReals')
        # PWA dependent flow of carrier into technology
        model.dependentFlowApproximation = pe.Var(
            model.setPWAConverEfficiency | model.setLinearConverEfficiency,
            domain = pe.NonNegativeReals,
            bounds = lambda _, tech, carrier, *__: cls.getAttributeOfAllElements("PWAParameter")[tech,"ConverEfficiency"]["bounds"][carrier],
            doc = 'PWA of flow of dependent carriers of conversion technologies. Dimensions: setPWAConverEfficiency | setLinearConverEfficiency. Domain: NonNegativeReals')

    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <ConversionTechnology> """
        model = EnergySystem.getConcreteModel()
        # add PWA constraints
        # capex
        if model.setPWACapex: 
            # if setPWACapex contains technologies:
            PWABreakpoints,PWAValues = cls.calculatePWABreakpointsValues(model.setPWACapex,"Capex")
            model.constraintPWACapex = pe.Piecewise(model.setPWACapex,
                model.capexApproximation,model.capacityApproximation,
                pw_pts = PWABreakpoints,pw_constr_type = "EQ", f_rule = PWAValues,unbounded_domain_var = True, warn_domain_coverage =False)
        if model.setLinearCapex:
            # if setLinearCapex contains technologies:
            model.constraintLinearCapex = pe.Constraint(
                model.setLinearCapex,
                rule = constraintLinearCapexRule,
                doc = "Linear relationship in capex. Dimension: setLinearCapex."
            )
        # Conversion Efficiency
        if model.setPWAConverEfficiency:
            # if setPWAConverEfficiency contains technologies:
            PWABreakpoints,PWAValues = cls.calculatePWABreakpointsValues(model.setPWAConverEfficiency,"ConverEfficiency")
            model.constraintPWAConverEfficiency = pe.Piecewise(model.setPWAConverEfficiency,
                model.dependentFlowApproximation,model.referenceFlowApproximation,
                pw_pts = PWABreakpoints,pw_constr_type = "EQ", f_rule = PWAValues,unbounded_domain_var = True, warn_domain_coverage =False)
        if model.setLinearConverEfficiency:
            # if setLinearConverEfficiency contains technologies:
            model.constraintLinearConverEfficiency = pe.Constraint(
                model.setLinearConverEfficiency,
                rule = constraintLinearConverEfficiencyRule,
                doc = "Linear relationship in ConverEfficiency. Dimension: setLinearConverEfficiency."
            )    
        ## Coupling constraints
        # couple the real variables with the auxiliary variables
        # capex
        model.constraintCapexCoupling = pe.Constraint(
            model.setPWACapex,
            rule = constraintCapexCouplingRule,
            doc = "couples the real capex variables with the approximated variables. Dimension: setPWACapex.")
        # capacity
        model.constraintCapacityCoupling = pe.Constraint(
            model.setPWACapex,
            rule = constraintCapacityCouplingRule,
            doc = "couples the real capacity variables with the approximated variables. Dimension: setPWACapex.")
    
    # defines disjuncts if technology on/off
    @classmethod
    def disjunctOnTechnologyRule(cls,disjunct, tech, node, time):
        """definition of disjunct constraints if technology is On"""
        model = disjunct.model()
        referenceCarrier = model.setReferenceCarriers[tech][1]
        if referenceCarrier in model.setInputCarriers[tech]:
            referenceFlow = model.inputFlow[tech,referenceCarrier,node,time]
        else:
            referenceFlow = model.outputFlow[tech,referenceCarrier,node,time]
        # disjunct constraints min load
        disjunct.constraintMinLoad = pe.Constraint(
            expr=referenceFlow >= model.minLoad[tech] * model.capacity[tech,node, time]
        )
        # couple reference flows
        disjunct.constraintReferenceFlowCoupling = pe.Constraint(
            [tech],
            model.setDependentCarriers[tech],
            [node],
            [time],
            rule = constraintReferenceFlowCouplingRule,
            doc = "couples the real reference flow variables with the approximated variables. Dimension: tech, setDependentCarriers[tech], node, time.")
        # couple dependent flows
        disjunct.constraintDependentFlowCoupling = pe.Constraint(
            [tech],
            model.setDependentCarriers[tech],
            [node],
            [time],
            rule = constraintDependentFlowCouplingRule,
            doc = "couples the real dependent flow variables with the approximated variables. Dimension: tech, setDependentCarriers[tech], node, time.")

    @classmethod
    def disjunctOffTechnologyRule(cls,disjunct, tech, node, time):
        """definition of disjunct constraints if technology is off"""
        model = disjunct.model()
        disjunct.constraintNoLoad = pe.Constraint(
            expr=
                sum(model.inputFlow[tech,inputCarrier,node,time]     for inputCarrier  in model.setInputCarriers[tech]) +
                sum(model.outputFlow[tech,outputCarrier,node,time]   for outputCarrier in model.setOutputCarriers[tech]) 
                == 0
        )
            
    @classmethod
    def calculatePWABreakpointsValues(cls,setPWA,typePWA):
        """ calculates the breakpoints and function values for piecewise affine constraint
        :param setPWA: set of variable indices in capex approximation, for which PWA is performed
        :param typePWA: variable, for which PWA is performed
        :return PWABreakpoints: dict of PWA breakpoint values
        :return PWAValues: dict of PWA function values """
        PWABreakpoints = {}
        PWAValues = {}

        # iterate through PWA variable's indexes
        for index in setPWA:
            PWABreakpoints[index] = []
            PWAValues[index] = []
            if setPWA.dimen > 1:
                tech = index[0]
            else:
                tech = index
            # retrieve PWA variables
            PWAParameter = cls.getAttributeOfAllElements("PWAParameter")
            if typePWA == "Capex":
                PWABreakpoints[index] = PWAParameter[(tech,typePWA)]["capacity"]
                PWAValues[index] = PWAParameter[(tech,typePWA)]["capex"]
            elif typePWA == "ConverEfficiency":
                PWABreakpoints[index] = PWAParameter[(tech,typePWA)][cls.getAttributeOfAllElements("referenceCarrier")[tech][0]]
                PWAValues[index] = PWAParameter[(tech,typePWA)][index[1]]

        return PWABreakpoints,PWAValues

### --- functions with constraint rules --- ###
def constraintLinearCapexRule(model,tech,node,time):
    """ if capacity and capex have a linear relationship"""
    return(model.capexApproximation[tech,node,time] == model.slopeLinearApproximationCapex[tech,node,time]*model.capacityApproximation[tech,node,time])

def constraintLinearConverEfficiencyRule(model,tech,dependentCarrier,node,time):
    """ if reference carrier and dependent carrier have a linear relationship"""
    return(
        model.dependentFlowApproximation[tech,dependentCarrier,node,time] 
        == model.slopeLinearApproximationConverEfficiency[tech,dependentCarrier, node,time]*model.referenceFlowApproximation[tech,dependentCarrier,node,time]
    )

def constraintCapexCouplingRule(model,tech,node,time):
    """ couples capex variables based on modeling technique"""
    return(model.capex[tech,node,time] == model.capexApproximation[tech,node,time])

def constraintCapacityCouplingRule(model,tech,node,time):
    """ couples capacity variables based on modeling technique"""
    return(model.builtCapacity[tech,node,time] == model.capacityApproximation[tech,node,time])

def constraintReferenceFlowCouplingRule(disjunct,tech,dependentCarrier,node,time):
    """ couples reference flow variables based on modeling technique"""
    model = disjunct.model()
    referenceCarrier = model.setReferenceCarriers[tech][1]
    if referenceCarrier in model.setInputCarriers[tech]:
        return(model.inputFlow[tech,referenceCarrier,node,time] == model.referenceFlowApproximation[tech,dependentCarrier,node,time])
    else:
        return(model.outputFlow[tech,referenceCarrier,node,time] == model.referenceFlowApproximation[tech,dependentCarrier,node,time])

def constraintDependentFlowCouplingRule(disjunct,tech,dependentCarrier,node,time):
    """ couples output flow variables based on modeling technique"""
    model = disjunct.model()
    if dependentCarrier in model.setInputCarriers[tech]:
        return(model.inputFlow[tech,dependentCarrier,node,time] == model.dependentFlowApproximation[tech,dependentCarrier,node,time])
    else:
        return(model.outputFlow[tech,dependentCarrier,node,time] == model.dependentFlowApproximation[tech,dependentCarrier,node,time])
    
#%% TODO implement conditioning for e.g. hydrogen