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
from pyomo.core.base import initializer
import pyomo.environ as pe
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
<<<<<<< HEAD
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
        _inputPath                              = paths["setConversionTechnologies"][self.name]["folder"]
        self.referenceCarrier                   = [self.dataInput.extractAttributeData(_inputPath,"referenceCarrier")]
        self.availability                       = self.dataInput.extractInputData(_inputPath,"availability",[indexNames["nameNodes"],indexNames["nameTimeSteps"]])
        # define input and output carrier
        self.inputCarrier,self.outputCarrier    = self.dataInput.extractConversionCarriers(_inputPath,self.referenceCarrier,"conversionBalanceConstant")
        # extract PWA parameters
        self.PWAParameter                       = self.dataInput.extractPWAData(_inputPath,self.name)

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to ConversionTechnology --- ###
    @classmethod
    def constructSets(cls):
        """ constructs the pe.Sets of the class <ConversionTechnology> """
        model = EnergySystem.getConcreteModel()
        # get input carriers
        _inputCarriers      = cls.getAttributeOfAllElements("inputCarrier")
        _outputCarriers     = cls.getAttributeOfAllElements("outputCarrier")
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
=======
        super().__init__(object, 'Conversion', tech)

        # %% Subsets
        subsets = {f'setInputCarriers{tech}':    f'Set of input carrier of {tech}. Subset: setInputCarriers',
                   f'setOutputCarriers{tech}':   f'Set of output carriers of {tech}. Subset: setOutputCarriers'}
        # merge new items with parameters dictionary from Technology class
        subsets = {**subsets, **self.getTechSubsets()}
        self.addSets(subsets)

        # %% Parameters
        params = {}
        # merge new items with parameters dictionary from Technology class
        params = {**params, **self.getTechParams()}
        self.addParams(params)

        # %% Variables
        variables = {
            f'input{tech}': f'Carrier input of {tech}. \
                            \n\t Dimensions: setInputCarriers{tech}, setNodes, setTimeSteps. \
                            \n\t Domain: NonNegativeReals',
            f'output{tech}': f'Carrier output {tech}. \
                             \n\t Dimensions: setOutputCarriers{tech}, setNodes, setTimeSteps. \
                             \n\t Domain: NonNegativeReals'}
        # merge new items with variables dictionary from Technology class
        variables = {**variables, **self.getTechVars()}
        self.addVars(variables)

        #%% Constraints
        constr = {
            f'{tech}MaxOutput': f'maximum output of {tech} is limited by the installed capacity. \
                                \n\t Dimensions: setOutputCarriers{tech}, setNodes, setTimeSteps'}
        constr = {**constr, **self.getTechConstr()}
        # add constraints defined in technology class
        self.addConstr(constr, replace = [tech, 'ConversionTechnology'], passValues = [tech])

        # add linear/nonlinear constraints to model capex and conversion efficiency
        for type, nonLinearTechs in self.analysis['nonlinearTechnologyApproximation'].items():
            if tech in nonLinearTechs:
                self.addNonlinearConstraints(type, tech)
            else:
                self.addLinearConstraints(type, tech)

        logging.info(f'added subsets, parameters, decision variables and constraints for {tech}')


    def addNonlinearConstraints(self, type, tech):
        """add subsets, parameters, variables, and constraints for nonlinear problem formulation
        :param tech: conversion technology
        :param type: parameter type of the nonlinear function (capex or efficiency)"""

        pass


    def addLinearConstraints(self, type, tech):
        """add subsets, parameters, variables, and constraints for linearized problem formulation
        :param tech: conversion technology
        :param type: type of the function that is linearized (capex or efficiency)"""

        #%% Subsets
        subsets = {f'setSegments{type}{tech}': f'Set of support points for PWA of {type} for {tech}'}
        self.addSets(subsets)

        #%% Parameters
        params = {f'slope{type}{tech}':     f'Parameter which specifies the slope of the {type} segment {tech}.\
                                            \n\t Dimensions: setSegments{type}{tech}',
                  f'intercept{type}{tech}': f'Parameter which specifies the intercept of the {type} segment {tech}.\
                                            \n\t Dimensions: setSegments{type}{tech}',
                  f'lbSegment{type}{tech}': f'Parameter which specifies the lower bound of the {type} segment {tech}.\
                                            \n\t Dimensions: setSegments{type}{tech}',
                  f'ubSegment{type}{tech}': f'Parameter which specifies the upper bound of the {type} segment {tech}.\
                                            \n\t Dimensions: setSegments{type}{tech}'}
        self.addParams(params)

        #%% Variables
        variables = {
            f'selectSegment{type}{tech}': f'Binary variable to model the activation of a segment in the PWA approximation of {type} of the {tech}. \
                                          \n\t Dimensions: setSegments{type}{tech}, setNodes, setTimeSteps.\
                                          \n\t Domain: Binary'}
        if type == 'Capex':
            variables[f'capacityAux{tech}'] = f'Auxiliary variable to model {type} of {tech} technologies. \
                                              \n\t Dimensions: setSegments{type}{tech}, setNodes, setTimeSteps.\
                                              \n\t Domain: NonNegativeReals'
        elif type == 'ConverEfficiency':
            variables[f'inputAux{tech}']   = f'Auxiliary variable to model {type} of {tech} technologies. \
                                             \n\t Dimensions: setSegments{type}{tech}, setInputCarriers{tech}, setNodes, setTimeSteps.\
                                             \n\t Domain: NonNegativeReals'
        self.addVars(variables)

        #%% Constraints
        constr = dict()
        if type == 'Capex':
            constr[f'{tech}Linear{type}']      = f'Linearization of {type} for {tech}.\
                                                 \n\t Dimensions:setNodes, setTimeSteps'
            constr[f'{tech}Linear{type}LB']    = f'lower bound of segment for {type} of {tech}.\
                                                 \n\t Dimensions: setSegments{type}{tech}, setNodes, setTimeSteps'
            constr[f'{tech}Linear{type}UB']    = f'upper bound of segment for {type} of {tech}.\
                                                 \n\t Dimensions: setSegments{type}{tech}, setNodes, setTimeSteps'
            constr[f'{tech}Linear{type}Aux']   = f'linking the auxiliary variable and variable for {type} of {tech}.\
                                                 \n\t Dimensions: setNodes, setTimeSteps'

        elif type == 'ConverEfficiency':
            constr[f'{tech}Linear{type}']    = f'Linearization of {type} for {type} of {tech}.\
                                               \n\t Dimensions: setInputCarriers{tech}, setOutputCarriers{tech}, setNodes, setTimeSteps'
            constr[f'{tech}Linear{type}LB']  = f'lower bound of segment for {type} of {tech}.\
                                               \n\t Dimensions: setSegments{type}{tech}, setInputCarriers{tech}, setNodes, setTimeSteps'
            constr[f'{tech}Linear{type}UB']  = f'upper bound of segment for {type} of {tech}.\
                                               \n\t Dimensions: setSegments{type}{tech}, setInputCarriers{tech}, setNodes, setTimeSteps'
            constr[f'{tech}Linear{type}Aux'] = f'linking the auxiliary variable and variable for {type} of {tech}.\
                                               \n\t Dimensions: setInputCarriers{tech}, setNodes, setTimeSteps'

        constr[f'{tech}Linear{type}SegmentSelection']  = f'Segment selection for {type} of {tech}.\
                                                         \n\t Dimensions: setNodes, setTimeSteps.'

        self.addConstr(constr, replace = [tech, 'ConversionTechnology'], passValues = [tech])

    #%% Constraint rules pre-defined in Technology class
    @staticmethod
    def constraintConversionTechnologyAvailabilityRule(model, tech, node, time):
        """limited availability of conversion technology"""

        # parameters
        availabilityTechnology = getattr(model, f'availability{tech}')
        # variables
        installTechnology      = getattr(model, f'install{tech}')

        return (availabilityTechnology[node, time] >= installTechnology[node, time])

    @staticmethod
    def constraintConversionTechnologyMinCapacityRule(model, tech, node, time):
        """min capacity of conversion technology"""

        # parameters
        minCapacityTechnology = getattr(model, f'minCapacity{tech}')
        # variables
        installTechnology     = getattr(model, f'install{tech}')
        capacityTechnology    = getattr(model, f'capacity{tech}')

        return (minCapacityTechnology * installTechnology[node, time]
                <= capacityTechnology[node, time])

    @staticmethod
    def constraintConversionTechnologyMaxCapacityRule(model, tech, node, time):
        """max capacity of conversion technology"""

        # parameters
        maxCapacityTechnology = getattr(model, f'maxCapacity{tech}')
        # variables
        installTechnology  = getattr(model, f'install{tech}')
        capacityTechnology = getattr(model, f'capacity{tech}')

        return(maxCapacityTechnology * installTechnology[node, time]
               >= capacityTechnology[node, time])

    @staticmethod
    def constraintConversionTechnologyLifetimeRule(model, tech, node, time):
        """limited lifetime of the technology"""

        # parameters
        lifetime = getattr(model, f'lifetime{tech}')
        # variables
        capacityTechnology = getattr(model, f'capacity{tech}')

        # time range
        t_start = max(1, t - lifetime + 1)
        t_end = time + 1

        return (capacityTechnology[node, time]
                == sum((capacityTechnology[node, t + 1] - capacityTechnology[node, t] for t in range(t_start, t_end))))

    @staticmethod
    def constraintConversionTechnologyMinCapacityExpansionRule(model, tech, node, time):
        """min capacity expansion of conversion technology"""

        # parameters
        minCapacityExpansion = getattr(model, f'minCapacityExpansion{tech}')
        # variables
        expandTechnology   = getattr(model, f'expand{tech}')
        capacityTechnology = getattr(model, f'capacity{tech}')

        return (expandTechnology[node, t] * minCapacityExpansion
                >= capacityTechnology[node, time] - capacityTechnology[node, time])

    @staticmethod
    def constraintConversionTechnologyMaxCapacityExpansionRule(model, tech, node, time):
        """max capacity expnsion of conversion technology"""

        # parameters
        maxCapacityExpansion = getattr(model, f'maxCapacityExpansion{tech}')
        # variables
        expandTechnology = getattr(model, f'expand{tech}')
        capacityTechnology = getattr(model, f'capacity{tech}')

        return (expandTechnology[node, t] * maxCapacityExpansion
                <= capacityTechnology[node, time] - capacityTechnology[node, time])

    @staticmethod
    def constraintConversionTechnologyLimitedCapacityExpansionRule(model, tech, node, time):
        """technology capacity can only be expanded once over its lifetime"""

        # parameters
        lifetime = getattr(model, f'lifetime{tech}')
        # variables
        expandTechnology = getattr(model, f'expand{tech}')

        # time range
        t_start = max(1, t - lifetime + 1)
        t_end = time + 1

        return (sum(expandTechnology[node, t] for t in range(t_start, t_end)) <= 1)
>>>>>>> development_v1_DT
        
        # set of PWA/NL technologies in capex/ConverEfficiency approximation
        model.setPWACapexTechs = pe.Set(
            initialize=[tech for tech in model.setConversionTechnologies if tech not in EnergySystem.getAnalysis()["nonlinearTechnologyApproximation"]["Capex"]],
            doc = "Set of conversion technologies for which the capex is PWA modeled")
        model.setNLCapexTechs = pe.Set(
            initialize=model.setConversionTechnologies - model.setPWACapexTechs,
            doc = "Set of conversion technologies for which the capex is NL modeled")
        model.setPWAConverEfficiencyTechs = pe.Set(
            initialize=[tech for tech in model.setConversionTechnologies if tech not in EnergySystem.getAnalysis()["nonlinearTechnologyApproximation"]["ConverEfficiency"]],
            doc = "Set of conversion technologies for which the ConverEfficiency is PWA modeled")
        model.setNLConverEfficiencyTechs = pe.Set(
            initialize=model.setConversionTechnologies - model.setPWAConverEfficiencyTechs,
            doc = "Set of conversion technologies for which the ConverEfficiency is NL modeled")
        # set of variable indices in capex/ConverEfficiency approximation
        model.setPWACapex = pe.Set(
            initialize = [(tech,node,timeStep)  for tech in model.setPWACapexTechs
                                                for node in model.setNodes
                                                for timeStep in model.setTimeSteps],
            doc = "Set of capex indices for which the capex is PWA modeled"
        )
        model.setNLCapex = pe.Set(
            initialize = [(tech,node,timeStep)  for tech in model.setNLCapexTechs
                                                for node in model.setNodes
                                                for timeStep in model.setTimeSteps],
            doc = "Set of capex indices for which the capex is nonlinearly modeled"
        )
        model.setPWAConverEfficiency = pe.Set(
            initialize = [(tech,inputCarrier,outputCarrier,node,timeStep) for tech in model.setPWAConverEfficiencyTechs 
                                                            for inputCarrier in model.setInputCarriers[tech] 
                                                            for outputCarrier in model.setOutputCarriers[tech]
                                                            for node in model.setNodes
                                                            for timeStep in model.setTimeSteps],
            doc = "Set of ConverEfficiency indices for which the ConverEfficiency is PWA modeled"
        )
        model.setNLConverEfficiency = pe.Set(
            initialize = [(tech,inputCarrier,outputCarrier,node,timeStep) for tech in model.setNLConverEfficiencyTechs 
                                                            for inputCarrier in model.setInputCarriers[tech] 
                                                            for outputCarrier in model.setOutputCarriers[tech]
                                                            for node in model.setNodes
                                                            for timeStep in model.setTimeSteps],
            doc = "Set of ConverEfficiency indices for which the ConverEfficiency is nonlinearly modeled"
        )
        
    @classmethod
    def constructParams(cls):
        """ constructs the pe.Params of the class <ConversionTechnology> """
        pass 
        
    @classmethod
    def constructVars(cls):
        """ constructs the pe.Vars of the class <ConversionTechnology> """
        model = EnergySystem.getConcreteModel()
        
        # input flow of carrier into technology
        model.inputFlow = pe.Var(
            model.setInputCarriersTechs,
            model.setNodes,
            model.setTimeSteps,
            domain = pe.NonNegativeReals,
            doc = 'Carrier input of conversion technologies. \n\t Dimensions: setInputCarriers, setNodes, setTimeSteps. \n\t Domain: NonNegativeReals'
        )
        # output flow of carrier into technology
        model.outputFlow = pe.Var(
            model.setOutputCarriersTechs,
            model.setNodes,
            model.setTimeSteps,
            domain = pe.NonNegativeReals,
            doc = 'Carrier output of conversion technologies. \n\t Dimensions: setOutputCarriers, setNodes, setTimeSteps. \n\t Domain: NonNegativeReals'
        )
        # PWA Variables for PWA
        # capex
        # PWA capacity
        model.capacityTechnologyPWA = pe.Var(
            model.setPWACapex,
            domain = pe.NonNegativeReals,
            doc = 'PWA variable for size of installed technology on edge i and time t.  \n\t Dimensions: setPWACapex.\n\t Domain: NonNegativeReals')
        # PWA capex technology
        model.capexPWA = pe.Var(
            model.setPWACapex,
            domain = pe.NonNegativeReals,
            doc = 'PWA variable for capex for installing technology on edge i and time t.  \n\t Dimensions: setPWACapex.\n\t Domain: NonNegativeReals')
        # ConvEfficiency
        # PWA input flow of carrier into technology
        model.inputFlowPWA = pe.Var(
            model.setPWAConverEfficiency,
            domain = pe.NonNegativeReals,
            doc = 'PWA Carrier input of conversion technologies. \n\t Dimensions: setPWAConverEfficiency. \n\t Domain: NonNegativeReals'
        )
        # PWA output flow of carrier into technology
        model.outputFlowPWA = pe.Var(
            model.setPWAConverEfficiency,
            domain = pe.NonNegativeReals,
            doc = 'PWA Carrier output of conversion technologies. \n\t Dimensions: setPWAConverEfficiency. \n\t Domain: NonNegativeReals'
        )
    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <ConversionTechnology> """
        model = EnergySystem.getConcreteModel()
        # maximum output flow 
        model.constraintConversionTechnologyMaxOutput = pe.Constraint(
            model.setOutputCarriersTechs,
            model.setNodes,
            model.setTimeSteps,
            rule = constraintConversionTechnologyMaxOutputRule,
            doc = 'maximum output of conversion technology is limited by the installed capacity. \n\t Dimensions: setOutputCarriers, setNodes, setTimeSteps'
        )
        # add PWA constraints
        # capex
        if model.setPWACapex: 
            # if setPWACapex contains technologies:
            PWABreakpoints,PWAValues = cls.calculatePWABreakpointsValues(model.setPWACapex,"Capex")
            model.constraintPWACapex = pe.Piecewise(model.setPWACapex,
                model.capexPWA,model.capacityTechnologyPWA,
                pw_pts = PWABreakpoints,pw_constr_type = "EQ", f_rule = PWAValues,unbounded_domain_var = True)
        # ConvEfficiency
        if model.setPWAConverEfficiency:
            # if setPWAConverEfficiency contains technologies:
            PWABreakpoints,PWAValues = cls.calculatePWABreakpointsValues(model.setPWAConverEfficiency,"ConverEfficiency")
            model.constraintPWAConverEfficiency = pe.Piecewise(model.setPWAConverEfficiency,
                model.outputFlowPWA,model.inputFlowPWA,
                pw_pts = PWABreakpoints,pw_constr_type = "EQ", f_rule = PWAValues,unbounded_domain_var = True)
        
        # Coupling constraints
        # couple the real variables with the modeled variables
        # capex
        model.constraintCapexCoupling = pe.Constraint(
            model.setConversionTechnologies,
            model.setNodes,
            model.setTimeSteps,
            rule = constraintCapexCouplingRule,
            doc = "couples the real capex variables with the modeled variables. \n\t Dimension: setConversionTechnologies, setNodes, setTimeSteps."
        )
        # capacity
        model.constraintCapacityCoupling = pe.Constraint(
            model.setConversionTechnologies,
            model.setNodes,
            model.setTimeSteps,
            rule = constraintCapacityCouplingRule,
            doc = "couples the real capacity variables with the modeled variables. \n\t Dimension: setConversionTechnologies, setNodes, setTimeSteps."
        )
        # inputFlow
        model.constraintInputFlowCoupling = pe.Constraint(
            model.setInputCarriersTechs,
            model.setNodes,
            model.setTimeSteps,
            rule = constraintInputFlowCouplingRule,
            doc = "couples the real inputFlow variables with the modeled variables. \n\t Dimension: setInputCarriers, setNodes, setTimeSteps."
        )
        # outputFlow
        model.constraintOutputFlowCoupling = pe.Constraint(
            model.setOutputCarriersTechs,
            model.setNodes,
            model.setTimeSteps,
            rule = constraintOutputFlowCouplingRule,
            doc = "couples the real outputFlow variables with the modeled variables. \n\t Dimension: setOutputCarriers, setNodes, setTimeSteps."
        )
    
    @classmethod
    def calculatePWABreakpointsValues(cls,setPWA,typePWA):
        """ calculates the breakpoints and function values for piecewise affine constraint
        :param setPWA: set of technologies, for which PWA is performed
        :param typePWA: variable, for which PWA is performed
        :return PWABreakpoints: dict of PWA breakpoint values
        :return PWAValues: dict of PWA function values """
        PWABreakpoints = {}
        PWAValues = {}

        # iterate through PWA technologies
        for index in setPWA:
            PWABreakpoints[index] = []
            PWAValues[index] = []
            if setPWA.dimen > 1:
                tech = index[0]
            else:
                tech = index
            # retrieve PWA variables
            PWAParameter = cls.getAttributeOfAllElements("PWAParameter")
            _slope = PWAParameter[(tech,typePWA)]["slope"]
            _intercept = PWAParameter[(tech,typePWA)]["intercept"]
            _ubSegment = PWAParameter[(tech,typePWA)]["ubSegment"]
            _lbSegment = PWAParameter[(tech,typePWA)]["lbSegment"]
            for _section in _slope:
                PWABreakpoints[index].append(_lbSegment[_section])
                PWAValues[index].append(_slope[_section]*_lbSegment[_section] + _intercept[_section])
            # last entry
            PWABreakpoints[index].append(_ubSegment[_section])
            PWAValues[index].append(_slope[_section]*_ubSegment[_section] + _intercept[_section])
        return PWABreakpoints,PWAValues

### --- functions with constraint rules --- ###
def constraintConversionTechnologyMaxOutputRule(model, tech, carrierOut, node, time):
    """output is limited by the installed capacity"""

    return (model.capacityTechnology[tech, node, time]
            >= model.outputFlow[tech,carrierOut, node, time]) # TODO: does not account for conversion efficiency of output

def constraintCapexCouplingRule(model,tech,node,time):
    """ couples capex variables based on modeling technique"""
    if tech in model.setPWACapexTechs:
        return(model.capex[tech,node,time] == model.capexPWA[tech,node,time])
    elif tech in model.setNLCapexTechs:
        logging.info("Nonlinear approximation of Capex not yet implemented, return Constraint.Skip for model.capex")
        return pe.Constraint.Skip

def constraintCapacityCouplingRule(model,tech,node,time):
    """ couples capacity variables based on modeling technique"""
    if tech in model.setPWACapexTechs:
        return(model.capacityTechnology[tech,node,time] == model.capacityTechnologyPWA[tech,node,time])
    elif tech in model.setNLCapexTechs:
        logging.info("Nonlinear approximation of Capex not yet implemented, return Constraint.Skip for model.capacityTechnology")
        return pe.Constraint.Skip

def constraintInputFlowCouplingRule(model,tech,inputCarrier,node,time):
    """ couples input flow variables based on modeling technique"""
    if tech in model.setPWAConverEfficiencyTechs:
        # TODO: currently only one input/output carrier per tech
        return(model.inputFlow[tech,inputCarrier,node,time] == model.inputFlowPWA[tech,inputCarrier,model.setOutputCarriers["electrolysis"][1],node,time])
    elif tech in model.setNLConverEfficiencyTechs:
        logging.info("Nonlinear approximation of ConverEfficiency not yet implemented, return Constraint.Skip for model.inputFlow")
        return pe.Constraint.Skip

def constraintOutputFlowCouplingRule(model,tech,outputCarrier,node,time):
    """ couples output flow variables based on modeling technique"""
    if tech in model.setPWAConverEfficiencyTechs:
        # TODO: currently only one input/output carrier per tech
        return(model.outputFlow[tech,outputCarrier,node,time] == model.outputFlowPWA[tech,model.setInputCarriers["electrolysis"][1],outputCarrier,node,time])
    elif tech in model.setNLConverEfficiencyTechs:
        logging.info("Nonlinear approximation of ConverEfficiency not yet implemented, return Constraint.Skip for model.outputFlow")
        return pe.Constraint.Skip

#%% TODO implement conditioning for e.g. hydrogen