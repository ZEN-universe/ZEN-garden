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
from ..component import Parameter, Variable, Constraint


class ConversionTechnology(Technology):
    # set label
    label = "set_conversion_technologies"
    location_type = "set_nodes"

    def __init__(self, tech, energy_system):
        """init conversion technology object
        :param tech: name of added technology
        :param energy_system: The energy system the element is part of"""

        logging.info(f'Initialize conversion technology {tech}')
        super().__init__(tech, energy_system)
        # store input data
        self.store_input_data()

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().store_input_data()
        # define input and output carrier
        self.input_carrier = self.datainput.extract_conversion_carriers()["input_carrier"]
        self.output_carrier = self.datainput.extract_conversion_carriers()["output_carrier"]
        self.energy_system.set_technology_of_carrier(self.name, self.input_carrier + self.output_carrier)
        # check if reference carrier in input and output carriers and set technology to correspondent carrier
        assert self.reference_carrier[0] in (
                    self.input_carrier + self.output_carrier), f"reference carrier {self.reference_carrier} of technology {self.name} not in input and output carriers {self.input_carrier + self.output_carrier}"
        # get conversion efficiency and capex
        self.get_conver_efficiency()
        self.get_annualized_capex()

    def get_conver_efficiency(self):
        """retrieves and stores conver_efficiency for <ConversionTechnology>.
        Each Child class overwrites method to store different conver_efficiency """
        # TODO read pwa Dict and set Params
        _pwa_conver_efficiency, self.conver_efficiency_is_pwa = self.datainput.extract_pwa_data("conver_efficiency")
        if self.conver_efficiency_is_pwa:
            self.pwa_conver_efficiency = _pwa_conver_efficiency
        else:
            self.conver_efficiency_linear = _pwa_conver_efficiency

    def get_annualized_capex(self):
        """ this method retrieves the total capex and converts it to annualized capex """
        _pwa_capex, self.capex_is_pwa = self.datainput.extract_pwa_data("capex")
        # annualize capex
        fractional_annuity = self.calculate_fractional_annuity()
        system = self.energy_system.system
        _fraction_year = system["unaggregated_time_steps_per_year"] / system["total_hours_per_year"]
        if not self.capex_is_pwa:
            self.capex_specific = _pwa_capex["capex"] * fractional_annuity + self.fixed_opex_specific * _fraction_year
        else:
            self.pwa_capex = _pwa_capex
            assert (self.fixed_opex_specific == self.fixed_opex_specific).all(), "pwa_capex is only implemented for constant values of fixed Opex"
            self.pwa_capex["capex"] = [(value * fractional_annuity + self.fixed_opex_specific[0] * _fraction_year) for value in self.pwa_capex["capex"]]
            # set bounds
            self.pwa_capex["bounds"]["capex"] = tuple([(bound * fractional_annuity + self.fixed_opex_specific[0] * _fraction_year) for bound in self.pwa_capex["bounds"]["capex"]])
        # calculate capex of existing capacity
        self.capex_existing_capacity = self.calculate_capex_of_existing_capacities()

    def calculate_capex_of_single_capacity(self, capacity, index):
        """ this method calculates the annualized capex of a single existing capacity. """
        if capacity == 0:
            return 0
        # linear
        if not self.capex_is_pwa:
            capex = self.capex_specific[index[0]].iloc[0] * capacity
        else:
            capex = np.interp(capacity, self.pwa_capex["capacity"], self.pwa_capex["capex"])
        return capex

    ### --- getter/setter classmethods
    @classmethod
    def get_capex_conver_efficiency_all_elements(cls, energy_system: EnergySystem, variable_type, selectPWA, index_names=None):
        """ similar to Element.get_attribute_of_all_elements but only for capex and conver_efficiency.
        If selectPWA, extract pwa attributes, otherwise linear.
        :param energy_system: The Energy system to add everything
        :param variable_type: either capex or conver_efficiency
        :param selectPWA: boolean if get attributes for pwa
        :return dict_of_attributes: returns dict of attribute values """
        _class_elements = energy_system.get_all_elements(cls)
        dict_of_attributes = {}
        if variable_type == "capex":
            _is_pwa_attribute = "capex_is_pwa"
            _attribute_name_pwa = "pwa_capex"
            _attribute_name_linear = "capex_specific"
        elif variable_type == "conver_efficiency":
            _is_pwa_attribute = "conver_efficiency_is_pwa"
            _attribute_name_pwa = "pwa_conver_efficiency"
            _attribute_name_linear = "conver_efficiency_linear"
        else:
            raise KeyError("Select either 'capex' or 'conver_efficiency'")
        for _element in _class_elements:
            # extract for pwa
            if getattr(_element, _is_pwa_attribute) and selectPWA:
                dict_of_attributes, _ = energy_system.append_attribute_of_element_to_dict(cls, _element, _attribute_name_pwa, dict_of_attributes)
            # extract for linear
            elif not getattr(_element, _is_pwa_attribute) and not selectPWA:
                dict_of_attributes, _ = energy_system.append_attribute_of_element_to_dict(cls, _element, _attribute_name_linear, dict_of_attributes)
            if not dict_of_attributes:
                _, index_names = cls.create_custom_set(index_names, energy_system)
                return (dict_of_attributes, index_names)
        dict_of_attributes = pd.concat(dict_of_attributes, keys=dict_of_attributes.keys())
        if not index_names:
            logging.warning(f"Initializing a parameter ({variable_type}) without the specifying the index names will be deprecated!")
            return dict_of_attributes
        else:
            custom_set, index_names = cls.create_custom_set(index_names, energy_system)
            dict_of_attributes = energy_system.check_for_subindex(dict_of_attributes, custom_set)
            return (dict_of_attributes, index_names)

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to ConversionTechnology --- ###
    @classmethod
    def construct_sets(cls, energy_system: EnergySystem):
        """ constructs the pe.Sets of the class <ConversionTechnology>
        :param energy_system: The Energy system to add everything"""
        model = energy_system.pyomo_model
        # get input carriers
        _input_carriers = energy_system.get_attribute_of_all_elements(cls, "input_carrier")
        _output_carriers = energy_system.get_attribute_of_all_elements(cls, "output_carrier")
        _reference_carrier = energy_system.get_attribute_of_all_elements(cls, "reference_carrier")
        _dependent_carriers = {}
        for tech in _input_carriers:
            _dependent_carriers[tech] = _input_carriers[tech] + _output_carriers[tech]
            _dependent_carriers[tech].remove(_reference_carrier[tech][0])
        # input carriers of technology
        model.set_input_carriers = pe.Set(model.set_conversion_technologies, initialize=_input_carriers,
            doc="set of carriers that are an input to a specific conversion technology. Dimensions: set_conversion_technologies")
        # output carriers of technology
        model.set_output_carriers = pe.Set(model.set_conversion_technologies, initialize=_output_carriers,
            doc="set of carriers that are an output to a specific conversion technology. Dimensions: set_conversion_technologies")
        # dependent carriers of technology
        model.set_dependent_carriers = pe.Set(model.set_conversion_technologies, initialize=_dependent_carriers,
            doc="set of carriers that are an output to a specific conversion technology.\n\t Dimensions: set_conversion_technologies")

        # add pe.Sets of the child classes
        for subclass in cls.__subclasses__():
            if np.size(energy_system.system[subclass.label]):
                subclass.construct_sets(energy_system)

    @classmethod
    def construct_params(cls, energy_system):
        """ constructs the pe.Params of the class <ConversionTechnology>
        :param energy_system: The Energy system to add everything"""
        # slope of linearly modeled capex
        energy_system.parameters.add_parameter(name="capex_specific_conversion",
            data=cls.get_capex_conver_efficiency_all_elements(energy_system, "capex", False, index_names=["set_conversion_technologies", "set_capex_linear", "set_nodes", "set_time_steps_yearly"]),
            doc="Parameter which specifies the slope of the capex if approximated linearly")
        # slope of linearly modeled conversion efficiencies
        energy_system.parameters.add_parameter(name="conver_efficiency_specific", data=cls.get_capex_conver_efficiency_all_elements(energy_system, "conver_efficiency", False,
                                                                                                                     index_names=["set_conversion_technologies", "set_conver_efficiency_linear",
                                                                                                                                  "set_nodes", "set_time_steps_yearly"]),
            doc="Parameter which specifies the slope of the conversion efficiency if approximated linearly")

    @classmethod
    def construct_vars(cls, energy_system: EnergySystem):
        """ constructs the pe.Vars of the class <ConversionTechnology>
        :param energy_system: The Energy system to add everything"""

        def carrier_flow_bounds(model, tech, carrier, node, time):
            """ return bounds of carrier_flow for bigM expression
            :param model: pe.ConcreteModel
            :param tech: tech index
            :param carrier: carrier index
            :param node: node index
            :param time: time index
            :return bounds: bounds of carrier_flow"""
            params = energy_system.parameters
            if energy_system.get_attribute_of_specific_element(cls, tech, "conver_efficiency_is_pwa"):
                bounds = energy_system.get_attribute_of_specific_element(cls, tech, "pwa_conver_efficiency")["bounds"][carrier]
            else:
                # convert operationTimeStep to time_step_year: operationTimeStep -> base_time_step -> time_step_year
                time_step_year = energy_system.convert_time_step_operation2invest(tech, time)
                if carrier == model.set_reference_carriers[tech].at(1):
                    _conver_efficiency = 1
                else:
                    _conver_efficiency = params.conver_efficiency_specific[tech, carrier, node, time_step_year]
                bounds = []
                for _bound in model.capacity[tech, "power", node, time_step_year].bounds:
                    if _bound is not None:
                        bounds.append(_bound * _conver_efficiency)
                    else:
                        bounds.append(None)
                bounds = tuple(bounds)
            return (bounds)

        model = energy_system.pyomo_model

        ## Flow variables
        # input flow of carrier into technology
        energy_system.variables.add_variable(model, name="input_flow", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_input_carriers", "set_nodes", "set_time_steps_operation"], energy_system),
            domain=pe.NonNegativeReals, bounds=carrier_flow_bounds, doc='Carrier input of conversion technologies')
        # output flow of carrier into technology
        energy_system.variables.add_variable(model, name="output_flow", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_output_carriers", "set_nodes", "set_time_steps_operation"], energy_system),
            domain=pe.NonNegativeReals, bounds=carrier_flow_bounds, doc='Carrier output of conversion technologies')

        ## pwa Variables - Capex
        # pwa capacity
        energy_system.variables.add_variable(model, name="capacity_approximation", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"], energy_system), domain=pe.NonNegativeReals,
            doc='pwa variable for size of installed technology on edge i and time t')
        # pwa capex technology
        energy_system.variables.add_variable(model, name="capex_approximation", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"], energy_system), domain=pe.NonNegativeReals,
            doc='pwa variable for capex for installing technology on edge i and time t')

        ## pwa Variables - Conversion Efficiency
        # pwa reference flow of carrier into technology
        energy_system.variables.add_variable(model, name="reference_flow_approximation",
            index_sets=cls.create_custom_set(["set_conversion_technologies", "set_dependent_carriers", "set_nodes", "set_time_steps_operation"], energy_system), domain=pe.NonNegativeReals,
            bounds=carrier_flow_bounds, doc='pwa of flow of reference carrier of conversion technologies')
        # pwa dependent flow of carrier into technology
        energy_system.variables.add_variable(model, name="dependent_flow_approximation",
            index_sets=cls.create_custom_set(["set_conversion_technologies", "set_dependent_carriers", "set_nodes", "set_time_steps_operation"], energy_system), domain=pe.NonNegativeReals,
            bounds=carrier_flow_bounds, doc='pwa of flow of dependent carriers of conversion technologies')

    @classmethod
    def construct_constraints(cls, energy_system):
        """ constructs the pe.Constraints of the class <ConversionTechnology>
        :param energy_system: The Energy system to add everything"""
        model = energy_system.pyomo_model
        # add pwa constraints
        # capex
        set_pwa_capex = cls.create_custom_set(["set_conversion_technologies", "set_capex_pwa", "set_nodes", "set_time_steps_yearly"], energy_system)
        set_linear_capex = cls.create_custom_set(["set_conversion_technologies", "set_capex_linear", "set_nodes", "set_time_steps_yearly"], energy_system)
        if set_pwa_capex:
            # if set_pwa_capex contains technologies:
            pwa_breakpoints, pwa_values = cls.calculate_pwa_breakpoints_values(energy_system, set_pwa_capex[0], "capex")
            model.constraint_pwa_capex = pe.Piecewise(set_pwa_capex[0], model.capex_approximation, model.capacity_approximation, pw_pts=pwa_breakpoints, pw_constr_type="EQ", f_rule=pwa_values,
                                                      unbounded_domain_var=True, warn_domain_coverage=False, pw_repn="BIGM_BIN")
        if set_linear_capex[0]:
            # if set_linear_capex contains technologies:
            energy_system.constraints.add_constraint(model, name="constraint_linear_capex", index_sets=set_linear_capex, rule=constraint_linear_capex_rule, doc="Linear relationship in capex")
        # Conversion Efficiency
        set_pwa_conver_efficiency = cls.create_custom_set(["set_conversion_technologies", "set_conver_efficiency_pwa", "set_nodes", "set_time_steps_operation"], energy_system)
        set_linear_conver_efficiency = cls.create_custom_set(["set_conversion_technologies", "set_conver_efficiency_linear", "set_nodes", "set_time_steps_operation"], energy_system)
        if set_pwa_conver_efficiency:
            # if set_pwa_conver_efficiency contains technologies:
            pwa_breakpoints, pwa_values = cls.calculate_pwa_breakpoints_values(energy_system, set_pwa_conver_efficiency[0], "conver_efficiency")
            model.constraint_pwa_conver_efficiency = pe.Piecewise(set_pwa_conver_efficiency[0], model.dependent_flow_approximation, model.reference_flow_approximation, pw_pts=pwa_breakpoints,
                                                                  pw_constr_type="EQ", f_rule=pwa_values, unbounded_domain_var=True, warn_domain_coverage=False, pw_repn="BIGM_BIN")
        if set_linear_conver_efficiency[0]:
            # if set_linear_conver_efficiency contains technologies:
            energy_system.constraints.add_constraint(model, name="constraint_linear_conver_efficiency", index_sets=set_linear_conver_efficiency, rule=constraint_linear_conver_efficiency_rule,
                doc="Linear relationship in conver_efficiency")  # Coupling constraints
        # couple the real variables with the auxiliary variables
        energy_system.constraints.add_constraint(model, name="constraint_capex_coupling", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"], energy_system),
            rule=constraint_capex_coupling_rule, doc="couples the real capex variables with the approximated variables")
        # capacity
        energy_system.constraints.add_constraint(model, name="constraint_capacity_coupling", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"], energy_system),
            rule=constraint_capacity_coupling_rule, doc="couples the real capacity variables with the approximated variables")

        # flow coupling constraints for technologies, which are not modeled with an on-off-behavior
        # reference flow coupling
        energy_system.constraints.add_constraint(model, name="constraint_reference_flow_coupling",
            index_sets=cls.create_custom_set(["set_conversion_technologies", "set_no_on_off", "set_dependent_carriers", "set_location", "set_time_steps_operation"], energy_system),
            rule=constraint_reference_flow_coupling_rule, doc="couples the real reference flow variables with the approximated variables")
        # dependent flow coupling
        energy_system.constraints.add_constraint(model, name="constraint_dependent_flow_coupling",
            index_sets=cls.create_custom_set(["set_conversion_technologies", "set_no_on_off", "set_dependent_carriers", "set_location", "set_time_steps_operation"], energy_system),
            rule=constraint_dependent_flow_coupling_rule, doc="couples the real dependent flow variables with the approximated variables")

    # defines disjuncts if technology on/off
    @classmethod
    def disjunct_on_technology_rule(cls, disjunct, tech, capacity_type, node, time):
        """definition of disjunct constraints if technology is On"""
        model = disjunct.model()
        # get parameter object
        params = Parameter.get_component_object()
        reference_carrier = model.set_reference_carriers[tech].at(1)
        if reference_carrier in model.set_input_carriers[tech]:
            reference_flow = model.input_flow[tech, reference_carrier, node, time]
        else:
            reference_flow = model.output_flow[tech, reference_carrier, node, time]
        # get invest time step
        time_step_year = EnergySystem.convert_time_step_operation2invest(tech, time)
        # disjunct constraints min load
        disjunct.constraint_min_load = pe.Constraint(expr=reference_flow >= params.min_load[tech, capacity_type, node, time] * model.capacity[tech, capacity_type, node, time_step_year])
        # couple reference flows
        Constraint.add_constraint(disjunct, name=f"constraint_reference_flow_coupling_{'_'.join([str(tech), str(node), str(time)])}",
            index_sets=[[[tech], model.set_dependent_carriers[tech], [node], [time]], ["set_conversion_technologies", "setDependentCarriers", "set_nodes", "set_time_steps_operation"]],
            rule=constraint_reference_flow_coupling_rule, doc="couples the real reference flow variables with the approximated variables", )
        # couple dependent flows
        Constraint.add_constraint(disjunct, name=f"constraint_dependent_flow_coupling_{'_'.join([str(tech), str(node), str(time)])}",
            index_sets=[[[tech], model.set_dependent_carriers[tech], [node], [time]], ["set_conversion_technologies", "setDependentCarriers", "set_nodes", "set_time_steps_operation"]],
            rule=constraint_dependent_flow_coupling_rule, doc="couples the real dependent flow variables with the approximated variables", )

    @classmethod
    def disjunct_off_technology_rule(cls, disjunct, tech, capacity_type, node, time):
        """definition of disjunct constraints if technology is off"""
        model = disjunct.model()
        disjunct.constraint_no_load = pe.Constraint(expr=sum(model.input_flow[tech, input_carrier, node, time] for input_carrier in model.set_input_carriers[tech]) + sum(
            model.output_flow[tech, output_carrier, node, time] for output_carrier in model.set_output_carriers[tech]) == 0)

    @classmethod
    def calculate_pwa_breakpoints_values(cls, energy_system: EnergySystem, setPWA, type_pwa):
        """ calculates the breakpoints and function values for piecewise affine constraint
        :param energy_system: The Energy system to add everything
        :param setPWA: set of variable indices in capex approximation, for which pwa is performed
        :param type_pwa: variable, for which pwa is performed
        :return pwa_breakpoints: dict of pwa breakpoint values
        :return pwa_values: dict of pwa function values """
        pwa_breakpoints = {}
        pwa_values = {}

        # iterate through pwa variable's indices
        for index in setPWA:
            pwa_breakpoints[index] = []
            pwa_values[index] = []
            if len(index) > 1:
                tech = index[0]
            else:
                tech = index
            if type_pwa == "capex":
                # retrieve pwa variables
                pwa_parameter = energy_system.get_attribute_of_specific_element(cls, tech, f"pwa_{type_pwa}")
                pwa_breakpoints[index] = pwa_parameter["capacity"]
                pwa_values[index] = pwa_parameter["capex"]
            elif type_pwa == "conver_efficiency":
                # retrieve pwa variables
                pwa_parameter = energy_system.get_attribute_of_specific_element(cls, tech, f"pwa_{type_pwa}")
                pwa_breakpoints[index] = pwa_parameter[energy_system.get_attribute_of_all_elements(cls, "reference_carrier")[tech][0]]
                pwa_values[index] = pwa_parameter[index[1]]

        return pwa_breakpoints, pwa_values


### --- functions with constraint rules --- ###
def constraint_linear_capex_rule(model, tech, node, time):
    """ if capacity and capex have a linear relationship"""
    # get parameter object
    params = Parameter.get_component_object()
    return (model.capex_approximation[tech, node, time] == params.capex_specific_conversion[tech, node, time] * model.capacity_approximation[tech, node, time])


def constraint_linear_conver_efficiency_rule(model, tech, dependent_carrier, node, time):
    """ if reference carrier and dependent carrier have a linear relationship"""
    # get parameter object
    params = Parameter.get_component_object()
    # get invest time step
    time_step_year = EnergySystem.convert_time_step_operation2invest(tech, time)
    return (model.dependent_flow_approximation[tech, dependent_carrier, node, time] == params.conver_efficiency_specific[tech, dependent_carrier, node, time_step_year] *
            model.reference_flow_approximation[tech, dependent_carrier, node, time])


def constraint_capex_coupling_rule(model, tech, node, time):
    """ couples capex variables based on modeling technique"""
    return (model.capex[tech, "power", node, time] == model.capex_approximation[tech, node, time])


def constraint_capacity_coupling_rule(model, tech, node, time):
    """ couples capacity variables based on modeling technique"""
    return (model.built_capacity[tech, "power", node, time] == model.capacity_approximation[tech, node, time])


def constraint_reference_flow_coupling_rule(disjunct, tech, dependent_carrier, node, time):
    """ couples reference flow variables based on modeling technique"""
    model = disjunct.model()
    reference_carrier = model.set_reference_carriers[tech].at(1)
    if reference_carrier in model.set_input_carriers[tech]:
        return (model.input_flow[tech, reference_carrier, node, time] == model.reference_flow_approximation[tech, dependent_carrier, node, time])
    else:
        return (model.output_flow[tech, reference_carrier, node, time] == model.reference_flow_approximation[tech, dependent_carrier, node, time])


def constraint_dependent_flow_coupling_rule(disjunct, tech, dependent_carrier, node, time):
    """ couples dependent flow variables based on modeling technique"""
    model = disjunct.model()
    if dependent_carrier in model.set_input_carriers[tech]:
        return (model.input_flow[tech, dependent_carrier, node, time] == model.dependent_flow_approximation[tech, dependent_carrier, node, time])
    else:
        return (model.output_flow[tech, dependent_carrier, node, time] == model.dependent_flow_approximation[tech, dependent_carrier, node, time])
