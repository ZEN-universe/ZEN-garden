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
        self.availability       = self.dataInput.extractInputData(_inputPath,"availability",[indexNames["nameNodes"],indexNames["nameTimeSteps"]])
        # define input and output carrier
        self.inputCarrier       = self.dataInput.extractConversionCarriers(_inputPath)["inputCarrier"]
        self.outputCarrier      = self.dataInput.extractConversionCarriers(_inputPath)["outputCarrier"]
        # extract PWA parameters
        self.PWAParameter       = self.dataInput.extractPWAData(_inputPath)

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to ConversionTechnology --- ###
    @classmethod
    def constructSets(cls):
        """ constructs the pe.Sets of the class <ConversionTechnology> """
        model = EnergySystem.getConcreteModel()
        # get input carriers
        _inputCarriers      = cls.getAttributeOfAllElements("inputCarrier")
        _outputCarriers     = cls.getAttributeOfAllElements("outputCarrier")
        _referenceCarrier   = cls.getAttributeOfAllElements("referenceCarrier")
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
        # set of variable indices in ConverEfficiency approximation
        model.setPWAConverEfficiency = pe.Set(
            initialize = [(tech,dependentCarrier,node,timeStep) for tech in model.setPWAConverEfficiencyTechs 
                                                            for dependentCarrier in model.setDependentCarriers[tech] 
                                                            for node in model.setNodes
                                                            for timeStep in model.setTimeSteps],
            doc = "Set of ConverEfficiency indices for which the ConverEfficiency is PWA modeled"
        )
        model.setNLConverEfficiency = pe.Set(
            initialize = [(tech,dependentCarrier,node,timeStep) for tech in model.setNLConverEfficiencyTechs 
                                                            for dependentCarrier in model.setDependentCarriers[tech] 
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
        
        ## Flow variables
        # input flow of carrier into technology
        model.inputFlow = pe.Var(
            model.setInputCarriersTechs,
            model.setNodes,
            model.setTimeSteps,
            domain = pe.NonNegativeReals,
            doc = 'Carrier input of conversion technologies. Dimensions: setConversionTechnologies, setInputCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals' )
        # output flow of carrier into technology
        model.outputFlow = pe.Var(
            model.setOutputCarriersTechs,
            model.setNodes,
            model.setTimeSteps,
            domain = pe.NonNegativeReals,
            doc = 'Carrier output of conversion technologies. Dimensions: setConversionTechnologies, setOutputCarriers, setNodes, setTimeSteps. Domain: NonNegativeReals')

        ## PWA Variables - Capex
        # PWA capacity
        model.capacityPWA = pe.Var(
            model.setPWACapex,
            domain = pe.NonNegativeReals,
            doc = 'PWA variable for size of installed technology on edge i and time t. Dimensions: setPWACapexTechs, setNodes, setTimeSteps. Domain: NonNegativeReals')
        # PWA capex technology
        model.capexPWA = pe.Var(
            model.setPWACapex,
            domain = pe.NonNegativeReals,
            doc = 'PWA variable for capex for installing technology on edge i and time t. Dimensions:  setPWACapexTechs, setNodes, setTimeSteps. Domain: NonNegativeReals')

        ## PWA Variables - Conversion Efficiency
        # PWA reference flow of carrier into technology
        model.referenceFlowPWA = pe.Var(
            model.setPWAConverEfficiency,
            domain = pe.NonNegativeReals,
            doc = 'PWA of flow of reference carrier of conversion technologies. Dimensions: setPWAConverEfficiencyTechs, setNodes, setTimeSteps. Domain: NonNegativeReals')
        # PWA dependent flow of carrier into technology
        model.dependentFlowPWA = pe.Var(
            model.setPWAConverEfficiency,
            domain = pe.NonNegativeReals,
            doc = 'PWA of flow of dependent carriers of conversion technologies. Dimensions: setPWAConverEfficiencyTechs, setNodes, setTimeSteps. Domain: NonNegativeReals')

    @classmethod
    def constructConstraints(cls):
        """ constructs the pe.Constraints of the class <ConversionTechnology> """
        model = EnergySystem.getConcreteModel()
        # maximum output flow 
        model.constraintConversionTechnologyMaxOutput = pe.Constraint(
            model.setConversionTechnologies,
            model.setNodes,
            model.setTimeSteps,
            rule = constraintConversionTechnologyMaxOutputRule,
            doc = 'maximum output of conversion technology is limited by the installed capacity. \n\t Dimensions: setConversionTechnologies, setNodes, setTimeSteps'
        )
        # link output carriers
        # model.constraintConversionTechnologyLinkOutputFlow = pe.Constraint(
        #     model.setOutputCarriersTechs,
        #     model.setNodes,
        #     model.setTimeSteps,
        #     rule = constraintConversionTechnologyLinkOutputFlowRule,
        #     doc = 'link the output carrier flow if conversion technology has multiple output carriers. Dimensions: setOutputCarriersTechs, setNodes, setTimeSteps')

        # add PWA constraints
        # capex
        if model.setPWACapex: 
            # if setPWACapex contains technologies:
            PWABreakpoints,PWAValues = cls.calculatePWABreakpointsValues(model.setPWACapex,"Capex")
            model.constraintPWACapex = pe.Piecewise(model.setPWACapex,
                model.capexPWA,model.capacityPWA,
                pw_pts = PWABreakpoints,pw_constr_type = "EQ", f_rule = PWAValues,unbounded_domain_var = True)
        # Conversion Efficiency
        if model.setPWAConverEfficiency:
            # if setPWAConverEfficiency contains technologies:
            PWABreakpoints,PWAValues = cls.calculatePWABreakpointsValues(model.setPWAConverEfficiency,"ConverEfficiency")
            model.constraintPWAConverEfficiency = pe.Piecewise(model.setPWAConverEfficiency,
                model.dependentFlowPWA,model.referenceFlowPWA,
                pw_pts = PWABreakpoints,pw_constr_type = "EQ", f_rule = PWAValues,unbounded_domain_var = True)
        
        ## Coupling constraints
        # couple the real variables with the auxiliary variables
        # capex
        model.constraintCapexCoupling = pe.Constraint(
            model.setConversionTechnologies,
            model.setNodes,
            model.setTimeSteps,
            rule = constraintCapexCouplingRule,
            doc = "couples the real capex variables with the approximated variables. Dimension: setConversionTechnologies, setNodes, setTimeSteps.")
        # capacity
        model.constraintCapacityCoupling = pe.Constraint(
            model.setConversionTechnologies,
            model.setNodes,
            model.setTimeSteps,
            rule = constraintCapacityCouplingRule,
            doc = "couples the real capacity variables with the approximated variables. Dimension: setConversionTechnologies, setNodes, setTimeSteps.")
        # reference flow
        model.constraintReferenceFlowCoupling = pe.Constraint(
            model.setPWAConverEfficiency,
            rule = constraintReferenceFlowCouplingRule,
            doc = "couples the real reference flow variables with the approximated variables. Dimension: setPWAConverEfficiency, setNodes, setTimeSteps.")
        # dependent flow
        model.constraintDependentFlowCoupling = pe.Constraint(
            model.setPWAConverEfficiency,
            rule = constraintDependentFlowCouplingRule,
            doc = "couples the real dependent flow variables with the approximated variables. Dimension: setPWAConverEfficiency, setNodes, setTimeSteps.")
    
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
def constraintConversionTechnologyMaxOutputRule(model, tech, node, time):
    """output is limited by the installed capacity"""
    referenceCarrier = model.setReferenceCarriers[tech][1]
    if referenceCarrier in model.setInputCarriers[tech]:
        return (model.capacity[tech, node, time] >= model.inputFlow[tech, referenceCarrier, node, time])
    else:
        return (model.capacity[tech, node, time] >= model.outputFlow[tech, referenceCarrier, node, time])

# def constraintConversionTechnologyLinkOutputFlowRule(model, tech, carrierOut, node, time):
#     """link output flow if technology has several output carriers"""
#     referenceCarrier = model.referenceCarrier[tech][0]
#     if carrierOut != referenceCarrier:
#         return (model.outputFlow[tech, referenceCarrier, node, time]
#                 == model.converEfficiency[tech, referenceCarrier, carrierOut] * model.outputFlow[tech, carrierOut, node, time])
#     else:
#         return pe.Constraint.Skip

def constraintCapexCouplingRule(model,tech,node,time):
    """ couples capex variables based on modeling technique"""
    if tech in model.setPWACapexTechs:
        return(model.capex[tech,node,time] == model.capexPWA[tech,node,time])
    elif tech in model.setNLCapexTechs:
        logging.info("Nonlinear approximation of Capex assigned by metaheuristic, return Constraint.Skip for model.capex")
        return pe.Constraint.Skip

def constraintCapacityCouplingRule(model,tech,node,time):
    """ couples capacity variables based on modeling technique"""
    if tech in model.setPWACapexTechs:
        return(model.capacity[tech,node,time] == model.capacityPWA[tech,node,time])
    elif tech in model.setNLCapexTechs:
        logging.info("Nonlinear approximation of Capex assigned by metaheuristic, return Constraint.Skip for model.capacity")
        return pe.Constraint.Skip

def constraintReferenceFlowCouplingRule(model,tech,dependentCarrier,node,time):
    """ couples reference flow variables based on modeling technique"""
    referenceCarrier = model.setReferenceCarriers[tech][1]
    if tech in model.setPWAConverEfficiencyTechs:
        if referenceCarrier in model.setInputCarriers[tech]:
            return(model.inputFlow[tech,referenceCarrier,node,time] == model.referenceFlowPWA[tech,dependentCarrier,node,time])
        else:
            return(model.outputFlow[tech,referenceCarrier,node,time] == model.referenceFlowPWA[tech,dependentCarrier,node,time])
    elif tech in model.setNLConverEfficiencyTechs:
        logging.info("Nonlinear approximation of efficiency not yet implemented, return Constraint.Skip for model.outputFlow[referenceCarrier]")

def constraintDependentFlowCouplingRule(model,tech,dependentCarrier,node,time):
    """ couples output flow variables based on modeling technique"""
    if tech in model.setPWAConverEfficiencyTechs:
        if dependentCarrier in model.setInputCarriers[tech]:
            return(model.inputFlow[tech,dependentCarrier,node,time] == model.dependentFlowPWA[tech,dependentCarrier,node,time])
        else:
            return(model.outputFlow[tech,dependentCarrier,node,time] == model.dependentFlowPWA[tech,dependentCarrier,node,time])
    elif tech in model.setNLConverEfficiencyTechs:
        logging.info("Nonlinear approximation of efficiency not yet implemented, return Constraint.Skip for model.outputFlow[dependentCarrier]")

#%% TODO implement conditioning for e.g. hydrogen