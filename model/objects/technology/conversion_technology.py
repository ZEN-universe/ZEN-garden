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
        self.deltaCapacity                      = self.dataInput.extractAttributeData(_inputPath,"deltaCapacity")
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
        model.capacityPWA = pe.Var(
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
                model.capexPWA,model.capacityPWA,
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

    return (model.capacity[tech, node, time]
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
        return(model.capacity[tech,node,time] == model.capacityPWA[tech,node,time])
    elif tech in model.setNLCapexTechs:
        logging.info("Nonlinear approximation of Capex not yet implemented, return Constraint.Skip for model.capacity")
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