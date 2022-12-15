"""===========================================================================================================================================================================
Title:          ZEN-GARDEN
Created:        October-2021
Authors:        Alissa Ganter (aganter@ethz.ch)
                Jacob Mannhardt (jmannhardt@ethz.ch)
Organization:   Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:    Class defining the parameters, variables and constraints of the conversion technologies.
                The class takes the abstract optimization model as an input, and adds parameters, variables and
                constraints of the conversion technologies.
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
import numpy as np
import pandas as pd
from .technology import Technology
from ..energy_system import EnergySystem
from ..component import Parameter,Variable,Constraint

class ConversionTechnology(Technology):
    # set label
    label           = "setConversionTechnologies"
    locationType    = "setNodes"
    # empty list of elements
    list_of_elements = []

    def __init__(self, tech):
        """init conversion technology object
        :param tech: name of added technology"""

        logging.info(f'Initialize conversion technology {tech}')
        super().__init__(tech)
        # store input data
        self.store_input_data()
        # add ConversionTechnology to list
        ConversionTechnology.add_element(self)

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """   
        # get attributes from class <Technology>
        super().store_input_data()
        # define input and output carrier
        self.inputCarrier               = self.datainput.extract_conversion_carriers()["inputCarrier"]
        self.outputCarrier              = self.datainput.extract_conversion_carriers()["outputCarrier"]
        EnergySystem.set_technology_of_carrier(self.name, self.inputCarrier + self.outputCarrier)
        # check if reference carrier in input and output carriers and set technology to correspondent carrier
        assert self.referenceCarrier[0] in (self.inputCarrier + self.outputCarrier), f"reference carrier {self.referenceCarrier} of technology {self.name} not in input and output carriers {self.inputCarrier + self.outputCarrier}"
        # get conversion efficiency and capex
        self.getConverEfficiency()
        self.getAnnualizedCapex()

    def getConverEfficiency(self):
        """retrieves and stores converEfficiency for <ConversionTechnology>.
        Each Child class overwrites method to store different converEfficiency """
        #TODO read pwa Dict and set Params
        _PWAConverEfficiency,self.converEfficiencyIsPWA   = self.datainput.extract_pwa_data("conver_efficiency")
        if self.converEfficiencyIsPWA:
            self.PWAConverEfficiency    = _PWAConverEfficiency
        else:
            self.conver_efficiency_linear = _PWAConverEfficiency

    def getAnnualizedCapex(self):
        """ this method retrieves the total capex and converts it to annualized capex """
        _PWACapex,self.capexIsPWA = self.datainput.extract_pwa_data("capex")
        # annualize capex
        fractionalAnnuity   = self.calculateFractionalAnnuity()
        system              = EnergySystem.get_system()
        _fraction_year     = system["unaggregatedTimeStepsPerYear"] / system["totalHoursPerYear"]
        if not self.capexIsPWA:
            self.capexSpecific = _PWACapex["capex"] * fractionalAnnuity + self.fixedOpexSpecific*_fraction_year
        else:
            self.PWACapex          = _PWACapex
            assert (self.fixedOpexSpecific==self.fixedOpexSpecific).all(), "PWACapex is only implemented for constant values of fixed Opex"
            self.PWACapex["capex"] = [(value * fractionalAnnuity + self.fixedOpexSpecific[0]*_fraction_year) for value in self.PWACapex["capex"]]
            # set bounds
            self.PWACapex["bounds"]["capex"] = tuple([(bound * fractionalAnnuity + self.fixedOpexSpecific[0]*_fraction_year) for bound in self.PWACapex["bounds"]["capex"]])
        # calculate capex of existing capacity
        self.capexExistingCapacity = self.calculateCapexOfExistingCapacities()

    def calculateCapexOfSingleCapacity(self,capacity,index):
        """ this method calculates the annualized capex of a single existing capacity. """
        if capacity == 0:
            return 0
        # linear
        if not self.capexIsPWA:
            capex   = self.capexSpecific[index[0]].iloc[0]*capacity
        else:
            capex   = np.interp(capacity,self.PWACapex["capacity"],self.PWACapex["capex"])
        return capex

    ### --- getter/setter classmethods
    @classmethod
    def getCapexConverEfficiencyOfAllElements(cls, variable_type, selectPWA,index_names = None):
        """ similar to Element.get_attribute_of_all_elements but only for capex and converEfficiency.
        If selectPWA, extract pwa attributes, otherwise linear.
        :param variable_type: either capex or converEfficiency
        :param selectPWA: boolean if get attributes for pwa
        :return dict_of_attributes: returns dict of attribute values """
        _class_elements      = cls.get_all_elements()
        dict_of_attributes    = {}
        if variable_type == "capex":
            _isPWAAttribute         = "capexIsPWA"
            _attributeNamePWA       = "PWACapex"
            _attributeNameLinear    = "capexSpecific"
        elif variable_type == "converEfficiency":
            _isPWAAttribute         = "converEfficiencyIsPWA"
            _attributeNamePWA       = "PWAConverEfficiency"
            _attributeNameLinear    = "conver_efficiency_linear"
        else:
            raise KeyError("Select either 'capex' or 'converEfficiency'")
        for _element in _class_elements:
            # extract for pwa
            if getattr(_element,_isPWAAttribute) and selectPWA:
                dict_of_attributes,_ = cls.append_attribute_of_element_to_dict(_element, _attributeNamePWA, dict_of_attributes)
            # extract for linear
            elif not getattr(_element,_isPWAAttribute) and not selectPWA:
                dict_of_attributes,_ = cls.append_attribute_of_element_to_dict(_element, _attributeNameLinear, dict_of_attributes)
            if not dict_of_attributes:
                _, index_names = cls.create_custom_set(index_names)
                return (dict_of_attributes,index_names)
        dict_of_attributes = pd.concat(dict_of_attributes,keys=dict_of_attributes.keys())
        if not index_names:
            logging.warning(f"Initializing a parameter ({variable_type}) without the specifying the index names will be deprecated!")
            return dict_of_attributes
        else:
            custom_set,index_names = cls.create_custom_set(index_names)
            dict_of_attributes    = EnergySystem.check_for_subindex(dict_of_attributes, custom_set)
            return (dict_of_attributes,index_names)

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to ConversionTechnology --- ###
    @classmethod
    def construct_sets(cls):
        """ constructs the pe.Sets of the class <ConversionTechnology> """
        model = EnergySystem.get_pyomo_model()
        # get input carriers
        _inputCarriers      = cls.get_attribute_of_all_elements("inputCarrier")
        _outputCarriers     = cls.get_attribute_of_all_elements("outputCarrier")
        _referenceCarrier   = cls.get_attribute_of_all_elements("referenceCarrier")
        _dependentCarriers  = {}
        for tech in _inputCarriers:
            _dependentCarriers[tech] = _inputCarriers[tech]+_outputCarriers[tech]
            _dependentCarriers[tech].remove(_referenceCarrier[tech][0])
        # input carriers of technology
        model.setInputCarriers = pe.Set(
            model.setConversionTechnologies,
            initialize = _inputCarriers,
            doc = "set of carriers that are an input to a specific conversion technology. Dimensions: setConversionTechnologies"
        )
        # output carriers of technology
        model.setOutputCarriers = pe.Set(
            model.setConversionTechnologies,
            initialize = _outputCarriers,
            doc = "set of carriers that are an output to a specific conversion technology. Dimensions: setConversionTechnologies"
        )
        # dependent carriers of technology
        model.setDependentCarriers = pe.Set(
            model.setConversionTechnologies,
            initialize = _dependentCarriers,
            doc = "set of carriers that are an output to a specific conversion technology.\n\t Dimensions: setConversionTechnologies"
        )

        # add pe.Sets of the child classes
        for subclass in cls.get_all_subclasses():
            if np.size(EnergySystem.get_system()[subclass.label]):
                subclass.construct_sets()

    @classmethod
    def construct_params(cls):
        """ constructs the pe.Params of the class <ConversionTechnology> """
        # slope of linearly modeled capex
        Parameter.add_parameter(
            name="capexSpecificConversion",
            data= cls.getCapexConverEfficiencyOfAllElements("capex",False,index_names=["setConversionTechnologies","set_capex_linear","setNodes","set_time_steps_yearly"]),
            doc = "Parameter which specifies the slope of the capex if approximated linearly"
        )
        # slope of linearly modeled conversion efficiencies
        Parameter.add_parameter(
            name="converEfficiencySpecific",
            data= cls.getCapexConverEfficiencyOfAllElements("converEfficiency",False,index_names=["setConversionTechnologies","set_conver_efficiency_linear","setNodes","set_time_steps_yearly"]),
            doc = "Parameter which specifies the slope of the conversion efficiency if approximated linearly"
        )

    @classmethod
    def construct_vars(cls):
        """ constructs the pe.Vars of the class <ConversionTechnology> """
        def carrierFlowBounds(model, tech, carrier, node, time):
            """ return bounds of carrierFlow for bigM expression
            :param model: pe.ConcreteModel
            :param tech: tech index
            :param carrier: carrier index
            :param node: node index
            :param time: time index
            :return bounds: bounds of carrierFlow"""
            params = Parameter.get_component_object()
            if cls.get_attribute_of_specific_element(tech,"converEfficiencyIsPWA"):
                bounds = cls.get_attribute_of_specific_element(tech,"PWAConverEfficiency")["bounds"][carrier]
            else:
                # convert operationTimeStep to timeStepYear: operationTimeStep -> base_time_step -> timeStepYear
                timeStepYear = EnergySystem.convert_time_step_operation2invest(tech,time)
                if carrier == model.setReferenceCarriers[tech].at(1):
                    _converEfficiency = 1
                else:
                    _converEfficiency = params.converEfficiencySpecific[tech,carrier,node,timeStepYear]
                bounds = []
                for _bound in model.capacity[tech, "power", node, timeStepYear].bounds:
                    if _bound is not None:
                        bounds.append(_bound*_converEfficiency)
                    else:
                        bounds.append(None)
                bounds = tuple(bounds)
            return (bounds)

        model = EnergySystem.get_pyomo_model()
        
        ## Flow variables
        # input flow of carrier into technology
        Variable.add_variable(
            model,
            name="inputFlow",
            index_sets= cls.create_custom_set(["setConversionTechnologies","setInputCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            bounds = carrierFlowBounds,
            doc = 'Carrier input of conversion technologies')
        # output flow of carrier into technology
        Variable.add_variable(
            model,
            name="outputFlow",
            index_sets= cls.create_custom_set(["setConversionTechnologies","setOutputCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            bounds = carrierFlowBounds,
            doc = 'Carrier output of conversion technologies')
        
        ## pwa Variables - Capex
        # pwa capacity
        Variable.add_variable(
            model,
            name="capacityApproximation",
            index_sets= cls.create_custom_set(["setConversionTechnologies","setNodes","set_time_steps_yearly"]),
            domain = pe.NonNegativeReals,
            doc = 'pwa variable for size of installed technology on edge i and time t')
        # pwa capex technology
        Variable.add_variable(
            model,
            name="capexApproximation",
            index_sets= cls.create_custom_set(["setConversionTechnologies","setNodes","set_time_steps_yearly"]),
            domain = pe.NonNegativeReals,
            doc = 'pwa variable for capex for installing technology on edge i and time t')

        ## pwa Variables - Conversion Efficiency
        # pwa reference flow of carrier into technology
        Variable.add_variable(
            model,
            name="referenceFlowApproximation",
            index_sets= cls.create_custom_set(["setConversionTechnologies","setDependentCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            bounds = carrierFlowBounds,
            doc = 'pwa of flow of reference carrier of conversion technologies')
        # pwa dependent flow of carrier into technology
        Variable.add_variable(
            model,
            name="dependentFlowApproximation",
            index_sets= cls.create_custom_set(["setConversionTechnologies","setDependentCarriers","setNodes","setTimeStepsOperation"]),
            domain = pe.NonNegativeReals,
            bounds = carrierFlowBounds,
            doc = 'pwa of flow of dependent carriers of conversion technologies')

    @classmethod
    def construct_constraints(cls):
        """ constructs the pe.Constraints of the class <ConversionTechnology> """
        model = EnergySystem.get_pyomo_model()
        # add pwa constraints
        # capex
        setPWACapex    = cls.create_custom_set(["setConversionTechnologies","set_capex_pwa","setNodes","set_time_steps_yearly"])
        setLinearCapex = cls.create_custom_set(["setConversionTechnologies","set_capex_linear","setNodes","set_time_steps_yearly"])
        if setPWACapex: 
            # if setPWACapex contains technologies:
            PWABreakpoints,PWAValues = cls.calculatePWABreakpointsValues(setPWACapex[0],"capex")
            model.constraintPWACapex = pe.Piecewise(setPWACapex[0],
                model.capexApproximation,model.capacityApproximation,
                pw_pts = PWABreakpoints,pw_constr_type = "EQ", f_rule = PWAValues,unbounded_domain_var = True, warn_domain_coverage =False,pw_repn="BIGM_BIN")
        if setLinearCapex[0]:
            # if setLinearCapex contains technologies:
            Constraint.add_constraint(
                model,
                name="constraintLinearCapex",
                index_sets= setLinearCapex,
                rule = constraintLinearCapexRule,
                doc = "Linear relationship in capex"
            )
        # Conversion Efficiency
        setPWAConverEfficiency      = cls.create_custom_set(["setConversionTechnologies","set_conver_efficiency_pwa","setNodes","setTimeStepsOperation"])
        setLinearConverEfficiency   = cls.create_custom_set(["setConversionTechnologies","set_conver_efficiency_linear","setNodes","setTimeStepsOperation"])
        if setPWAConverEfficiency:
            # if setPWAConverEfficiency contains technologies:
            PWABreakpoints,PWAValues = cls.calculatePWABreakpointsValues(setPWAConverEfficiency[0],"conver_efficiency")
            model.constraintPWAConverEfficiency = pe.Piecewise(setPWAConverEfficiency[0],
                model.dependentFlowApproximation,model.referenceFlowApproximation,
                pw_pts = PWABreakpoints,pw_constr_type = "EQ", f_rule = PWAValues,unbounded_domain_var = True, warn_domain_coverage =False,pw_repn="BIGM_BIN")
        if setLinearConverEfficiency[0]:
            # if setLinearConverEfficiency contains technologies:
            Constraint.add_constraint(
                model,
                name="constraintLinearConverEfficiency",
                index_sets= setLinearConverEfficiency,
                rule = constraintLinearConverEfficiencyRule,
                doc = "Linear relationship in ConverEfficiency"
            )    
        # Coupling constraints
        # couple the real variables with the auxiliary variables
        Constraint.add_constraint(
            model,
            name="constraintCapexCoupling",
            index_sets= cls.create_custom_set(["setConversionTechnologies","setNodes","set_time_steps_yearly"]),
            rule = constraintCapexCouplingRule,
            doc = "couples the real capex variables with the approximated variables")
        # capacity
        Constraint.add_constraint(
            model,
            name="constraintCapacityCoupling",
            index_sets= cls.create_custom_set(["setConversionTechnologies","setNodes","set_time_steps_yearly"]),
            rule = constraintCapacityCouplingRule,
            doc = "couples the real capacity variables with the approximated variables")
        
        # flow coupling constraints for technologies, which are not modeled with an on-off-behavior
        # reference flow coupling
        Constraint.add_constraint(
            model,
            name="constraintReferenceFlowCoupling",
            index_sets= cls.create_custom_set(["setConversionTechnologies","set_no_on_off","setDependentCarriers","setLocation","setTimeStepsOperation"]),
            rule = constraintReferenceFlowCouplingRule,
            doc = "couples the real reference flow variables with the approximated variables")
        # dependent flow coupling
        Constraint.add_constraint(
            model,
            name="constraintDependentFlowCoupling",
            index_sets= cls.create_custom_set(["setConversionTechnologies","set_no_on_off","setDependentCarriers","setLocation","setTimeStepsOperation"]),
            rule = constraintDependentFlowCouplingRule,
            doc = "couples the real dependent flow variables with the approximated variables")

    # defines disjuncts if technology on/off
    @classmethod
    def disjunctOnTechnologyRule(cls,disjunct, tech,capacity_type, node, time):
        """definition of disjunct constraints if technology is On"""
        model = disjunct.model()
        # get parameter object
        params = Parameter.get_component_object()
        referenceCarrier = model.setReferenceCarriers[tech].at(1)
        if referenceCarrier in model.setInputCarriers[tech]:
            referenceFlow = model.inputFlow[tech,referenceCarrier,node,time]
        else:
            referenceFlow = model.outputFlow[tech,referenceCarrier,node,time]
        # get invest time step
        timeStepYear = EnergySystem.convert_time_step_operation2invest(tech,time)
        # disjunct constraints min load
        disjunct.constraintMinLoad = pe.Constraint(
            expr=referenceFlow >= params.minLoad[tech,capacity_type,node,time] * model.capacity[tech,capacity_type,node, timeStepYear]
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
    def disjunctOffTechnologyRule(cls,disjunct, tech, capacity_type, node, time):
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
        :param setPWA: set of variable indices in capex approximation, for which pwa is performed
        :param typePWA: variable, for which pwa is performed
        :return PWABreakpoints: dict of pwa breakpoint values
        :return PWAValues: dict of pwa function values """
        PWABreakpoints = {}
        PWAValues = {}

        # iterate through pwa variable's indices
        for index in setPWA:
            PWABreakpoints[index] = []
            PWAValues[index] = []
            if len(index) > 1:
                tech = index[0]
            else:
                tech = index
            # retrieve pwa variables
            PWAParameter = cls.get_attribute_of_specific_element(tech,f"pwa{typePWA}")
            if typePWA == "capex":
                PWABreakpoints[index] = PWAParameter["capacity"]
                PWAValues[index] = PWAParameter["capex"]
            elif typePWA == "conver_efficiency":
                PWABreakpoints[index] = PWAParameter[cls.get_attribute_of_all_elements("referenceCarrier")[tech][0]]
                PWAValues[index] = PWAParameter[index[1]]

        return PWABreakpoints,PWAValues

### --- functions with constraint rules --- ###
def constraintLinearCapexRule(model,tech,node,time):
    """ if capacity and capex have a linear relationship"""
    # get parameter object
    params = Parameter.get_component_object()
    return(model.capexApproximation[tech,node,time] == params.capexSpecificConversion[tech,node,time]*model.capacityApproximation[tech,node,time])

def constraintLinearConverEfficiencyRule(model,tech,dependent_carrier,node,time):
    """ if reference carrier and dependent carrier have a linear relationship"""
    # get parameter object
    params = Parameter.get_component_object()
    # get invest time step
    timeStepYear = EnergySystem.convert_time_step_operation2invest(tech,time)
    return(
        model.dependentFlowApproximation[tech,dependent_carrier,node,time]
        == params.converEfficiencySpecific[tech,dependent_carrier, node,timeStepYear]*model.referenceFlowApproximation[tech,dependent_carrier,node,time]
    )

def constraintCapexCouplingRule(model,tech,node,time):
    """ couples capex variables based on modeling technique"""
    return(model.capex[tech,"power",node,time] == model.capexApproximation[tech,node,time])

def constraintCapacityCouplingRule(model,tech,node,time):
    """ couples capacity variables based on modeling technique"""
    return(model.built_capacity[tech,"power",node,time] == model.capacityApproximation[tech,node,time])

def constraintReferenceFlowCouplingRule(disjunct,tech,dependent_carrier,node,time):
    """ couples reference flow variables based on modeling technique"""
    model = disjunct.model()
    referenceCarrier = model.setReferenceCarriers[tech].at(1)
    if referenceCarrier in model.setInputCarriers[tech]:
        return(model.inputFlow[tech,referenceCarrier,node,time] == model.referenceFlowApproximation[tech,dependent_carrier,node,time])
    else:
        return(model.outputFlow[tech,referenceCarrier,node,time] == model.referenceFlowApproximation[tech,dependent_carrier,node,time])

def constraintDependentFlowCouplingRule(disjunct,tech,dependent_carrier,node,time):
    """ couples output flow variables based on modeling technique"""
    model = disjunct.model()
    if dependent_carrier in model.setInputCarriers[tech]:
        return(model.inputFlow[tech,dependent_carrier,node,time] == model.dependentFlowApproximation[tech,dependent_carrier,node,time])
    else:
        return(model.outputFlow[tech,dependent_carrier,node,time] == model.dependentFlowApproximation[tech,dependent_carrier,node,time])
