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
import time

import numpy as np
import pandas as pd
import xarray as xr

from zen_garden.utils import linexpr_from_tuple_np
from .technology import Technology
from ..component import ZenIndex


class ConversionTechnology(Technology):
    # set label
    label = "set_conversion_technologies"
    location_type = "set_nodes"

    def __init__(self, tech, optimization_setup):
        """init conversion technology object
        :param tech: name of added technology
        :param optimization_setup: The OptimizationSetup the element is part of """

        logging.info(f'Initialize conversion technology {tech}')
        super().__init__(tech, optimization_setup)
        # store input data
        self.store_input_data()

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().store_input_data()
        # define input and output carrier
        self.input_carrier = self.data_input.extract_conversion_carriers()["input_carrier"]
        self.output_carrier = self.data_input.extract_conversion_carriers()["output_carrier"]
        self.energy_system.set_technology_of_carrier(self.name, self.input_carrier + self.output_carrier)
        # check if reference carrier in input and output carriers and set technology to correspondent carrier
        assert self.reference_carrier[0] in (self.input_carrier + self.output_carrier), \
            f"reference carrier {self.reference_carrier} of technology {self.name} not in input and output carriers {self.input_carrier + self.output_carrier}"
        # get conversion efficiency and capex
        self.get_conver_efficiency()
        self.convert_to_fraction_of_capex()

    def get_conver_efficiency(self):
        """retrieves and stores conver_efficiency for <ConversionTechnology>.
        Each Child class overwrites method to store different conver_efficiency """
        # TODO read pwa Dict and set Params
        _pwa_conver_efficiency, self.conver_efficiency_is_pwa = self.data_input.extract_pwa_data("conver_efficiency")
        if self.conver_efficiency_is_pwa:
            self.pwa_conver_efficiency = _pwa_conver_efficiency
        else:
            self.conver_efficiency_linear = _pwa_conver_efficiency

    def convert_to_fraction_of_capex(self):
        """ this method retrieves the total capex and converts it to annualized capex """
        _pwa_capex, self.capex_is_pwa = self.data_input.extract_pwa_data("capex")
        # annualize capex
        fraction_year = self.calculate_fraction_of_year()
        self.fixed_opex_specific = self.fixed_opex_specific * fraction_year
        if not self.capex_is_pwa:
            self.capex_specific = _pwa_capex["capex"] * fraction_year
        else:
            self.pwa_capex = _pwa_capex
            self.pwa_capex["capex"] = [value * fraction_year for value in self.pwa_capex["capex"]]
            # set bounds
            self.pwa_capex["bounds"]["capex"] = tuple([(bound * fraction_year) for bound in self.pwa_capex["bounds"]["capex"]])
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
    def get_capex_conver_efficiency_all_elements(cls, optimization_setup, variable_type, selectPWA, index_names=None):
        """ similar to Element.get_attribute_of_all_elements but only for capex and conver_efficiency.
        If selectPWA, extract pwa attributes, otherwise linear.
        :param optimization_setup: The OptimizationSetup the element is part of
        :param variable_type: either capex or conver_efficiency
        :param selectPWA: boolean if get attributes for pwa
        :return dict_of_attributes: returns dict of attribute values """
        _class_elements = optimization_setup.get_all_elements(cls)
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
                dict_of_attributes, _ = optimization_setup.append_attribute_of_element_to_dict(_element, _attribute_name_pwa, dict_of_attributes)
            # extract for linear
            elif not getattr(_element, _is_pwa_attribute) and not selectPWA:
                dict_of_attributes, _ = optimization_setup.append_attribute_of_element_to_dict(_element, _attribute_name_linear, dict_of_attributes)
            if not dict_of_attributes:
                _, index_names = cls.create_custom_set(index_names, optimization_setup)
                return (dict_of_attributes, index_names)
        dict_of_attributes = pd.concat(dict_of_attributes, keys=dict_of_attributes.keys())
        if not index_names:
            logging.warning(f"Initializing a parameter ({variable_type}) without the specifying the index names will be deprecated!")
            return dict_of_attributes
        else:
            custom_set, index_names = cls.create_custom_set(index_names, optimization_setup)
            dict_of_attributes = optimization_setup.check_for_subindex(dict_of_attributes, custom_set)
            return (dict_of_attributes, index_names)

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to ConversionTechnology --- ###
    @classmethod
    def construct_sets(cls, optimization_setup):
        """ constructs the pe.Sets of the class <ConversionTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        # get input carriers
        _input_carriers = optimization_setup.get_attribute_of_all_elements(cls, "input_carrier")
        _output_carriers = optimization_setup.get_attribute_of_all_elements(cls, "output_carrier")
        _reference_carrier = optimization_setup.get_attribute_of_all_elements(cls, "reference_carrier")
        _dependent_carriers = {}
        for tech in _input_carriers:
            _dependent_carriers[tech] = _input_carriers[tech] + _output_carriers[tech]
            _dependent_carriers[tech].remove(_reference_carrier[tech][0])
        # input carriers of technology
        optimization_setup.sets.add_set(name="set_input_carriers", data=_input_carriers,
                                        doc="set of carriers that are an input to a specific conversion technology. Dimensions: set_conversion_technologies",
                                        index_set="set_conversion_technologies")
        # output carriers of technology
        optimization_setup.sets.add_set(name="set_output_carriers", data=_output_carriers,
                                        doc="set of carriers that are an output to a specific conversion technology. Dimensions: set_conversion_technologies",
                                        index_set="set_conversion_technologies")
        # dependent carriers of technology
        optimization_setup.sets.add_set(name="set_dependent_carriers", data=_dependent_carriers,
                                        doc="set of carriers that are an output to a specific conversion technology.\n\t Dimensions: set_conversion_technologies",
                                        index_set="set_conversion_technologies")

        # add pe.Sets of the child classes
        for subclass in cls.__subclasses__():
            if np.size(optimization_setup.system[subclass.label]):
                subclass.construct_sets(optimization_setup)

    @classmethod
    def construct_params(cls, optimization_setup):
        """ constructs the pe.Params of the class <ConversionTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """
        # slope of linearly modeled capex
        optimization_setup.parameters.add_parameter(name="capex_specific_conversion",
            data=cls.get_capex_conver_efficiency_all_elements(optimization_setup, "capex", False, index_names=["set_conversion_technologies", "set_capex_linear", "set_nodes", "set_time_steps_yearly"]),
            doc="Parameter which specifies the slope of the capex if approximated linearly")
        # slope of linearly modeled conversion efficiencies
        optimization_setup.parameters.add_parameter(name="conver_efficiency_specific", data=cls.get_capex_conver_efficiency_all_elements(optimization_setup, "conver_efficiency", False,
                                                                                                                     index_names=["set_conversion_technologies", "set_conver_efficiency_linear",
                                                                                                                                  "set_nodes", "set_time_steps_yearly"]),
            doc="Parameter which specifies the slope of the conversion efficiency if approximated linearly")

    @classmethod
    def construct_vars(cls, optimization_setup):
        """ constructs the pe.Vars of the class <ConversionTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """

        model = optimization_setup.model
        variables = optimization_setup.variables

        def get_carrier_flow_bounds(index_values, index_names):
            """ return bounds of carrier_flow for bigM expression
            :param index_values: list of index values
            :return bounds: bounds of carrier_flow"""
            params = optimization_setup.parameters
            sets = optimization_setup.sets
            energy_system = optimization_setup.energy_system

            # init the bounds
            index_arrs = sets.tuple_to_arr(index_values, index_names)
            coords = [np.unique(t.data) for t in index_arrs]
            lower = xr.DataArray(0.0, coords=coords)
            upper = xr.DataArray(np.inf, coords=coords)

            # get the sets
            technology_set, carrier_set, node_set, timestep_set = [sets[name] for name in index_names]

            for tech in technology_set:
                for carrier in carrier_set[tech]:
                    if optimization_setup.get_attribute_of_specific_element(cls, tech, "conver_efficiency_is_pwa"):
                        bounds = optimization_setup.get_attribute_of_specific_element(cls, tech, "pwa_conver_efficiency")["bounds"][carrier]
                        # make sure lower bound is above 0
                        if bounds[0] is not None and bounds[0] > 0:
                            lower.loc[tech, carrier, ...] = bounds[0]
                        upper.loc[tech, carrier, ...] = bounds[1]
                    else:
                        time_step_year = energy_system.time_steps.convert_time_step_operation2year(tech, timestep_set[tech]).values
                        if carrier == sets["set_reference_carriers"][tech][0]:
                            _conver_efficiency = 1
                        else:
                            _conver_efficiency = params.conver_efficiency_specific.loc[tech, carrier, node_set, time_step_year]
                        lower.loc[tech, carrier, ...] = model.variables["capacity"].lower.loc[tech, "power", node_set, time_step_year].data
                        upper.loc[tech, carrier, ...] = model.variables["capacity"].upper.loc[tech, "power", node_set, time_step_year].data

            # make sure lower is never below 0
            return (lower, upper)

        ## Flow variables
        # input flow of carrier into technology
        index_values, index_names = cls.create_custom_set(["set_conversion_technologies", "set_input_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup)
        variables.add_variable(model, name="input_flow", index_sets=(index_values, index_names),
            bounds=get_carrier_flow_bounds(index_values, index_names), doc='Carrier input of conversion technologies')
        # output flow of carrier into technology
        index_values, index_names = cls.create_custom_set(["set_conversion_technologies", "set_output_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup)
        variables.add_variable(model, name="output_flow", index_sets=(index_values, index_names),
            bounds=get_carrier_flow_bounds(index_values, index_names), doc='Carrier output of conversion technologies')

        ## pwa Variables - Capex
        # pwa capacity
        variables.add_variable(model, name="capacity_approximation", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"], optimization_setup), bounds=(0,np.inf),
            doc='pwa variable for size of installed technology on edge i and time t')
        # pwa capex technology
        variables.add_variable(model, name="capex_approximation", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"], optimization_setup), bounds=(0,np.inf),
            doc='pwa variable for capex for installing technology on edge i and time t')

        ## pwa Variables - Conversion Efficiency
        # pwa reference flow of carrier into technology
        index_values, index_names = cls.create_custom_set(["set_conversion_technologies", "set_dependent_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup)
        variables.add_variable(model, name="reference_flow_approximation",
            index_sets=(index_values, index_names),
            bounds=get_carrier_flow_bounds(index_values, index_names), doc='pwa of flow of reference carrier of conversion technologies')
        # pwa dependent flow of carrier into technology
        index_values, index_names = cls.create_custom_set(["set_conversion_technologies", "set_dependent_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup)
        variables.add_variable(model, name="dependent_flow_approximation",
            index_sets=(index_values, index_names),
            bounds=get_carrier_flow_bounds(index_values, index_names), doc='pwa of flow of dependent carriers of conversion technologies')

    @classmethod
    def construct_constraints(cls, optimization_setup):
        """ constructs the pe.Constraints of the class <ConversionTechnology>
        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        constraints = optimization_setup.constraints
        # add pwa constraints
        rules = ConversionTechnologyRules(optimization_setup)
        # capex
        set_pwa_capex = cls.create_custom_set(["set_conversion_technologies", "set_capex_pwa", "set_nodes", "set_time_steps_yearly"], optimization_setup)
        set_linear_capex = cls.create_custom_set(["set_conversion_technologies", "set_capex_linear", "set_nodes", "set_time_steps_yearly"], optimization_setup)
        if len(set_pwa_capex[0]) > 0:
            # if set_pwa_capex contains technologies:
            pwa_breakpoints, pwa_values = cls.calculate_pwa_breakpoints_values(optimization_setup, set_pwa_capex[0], "capex")
            constraints.add_pw_constraint(model, index_values=set_pwa_capex[0], yvar="capex_approximation", xvar="capacity_approximation",
                                          break_points=pwa_breakpoints, f_vals=pwa_values, cons_type="EQ")
        if set_linear_capex[0]:
            # if set_linear_capex contains technologies:
            constraints.add_constraint_rule(model, name="constraint_linear_capex", index_sets=set_linear_capex, rule=rules.constraint_linear_capex_rule, doc="Linear relationship in capex")
        # Conversion Efficiency
        set_pwa_conver_efficiency = cls.create_custom_set(["set_conversion_technologies", "set_conver_efficiency_pwa", "set_nodes", "set_time_steps_operation"], optimization_setup)
        set_linear_conver_efficiency = cls.create_custom_set(["set_conversion_technologies", "set_conver_efficiency_linear", "set_nodes", "set_time_steps_operation"], optimization_setup)
        if len(set_pwa_conver_efficiency[0]) > 0:
            # if set_pwa_conver_efficiency contains technologies:
            pwa_breakpoints, pwa_values = cls.calculate_pwa_breakpoints_values(optimization_setup, set_pwa_conver_efficiency[0], "conver_efficiency")
            constraints.add_pw_constraint(model, index_values=set_pwa_conver_efficiency[0], yvar="dependent_flow_approximation", xvar="reference_flow_approximation",
                                          break_points=pwa_breakpoints, f_vals=pwa_values, cons_type="EQ")
        if set_linear_conver_efficiency[0]:
            # if set_linear_conver_efficiency contains technologies:
            constraints.add_constraint_block(model, name="constraint_linear_conver_efficiency",
                                             constraint=rules.get_constraint_linear_conver_efficiency(*set_linear_conver_efficiency),
                                             doc="Linear relationship in conver_efficiency")  # Coupling constraints
        # couple the real variables with the auxiliary variables
        t0 = time.perf_counter()
        constraints.add_constraint_rule(model, name="constraint_capex_coupling", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"], optimization_setup),
            rule=rules.constraint_capex_coupling_rule, doc="couples the real capex variables with the approximated variables")
        t1 = time.perf_counter()
        logging.debug(f"Conversion Technology: constraint_capex_coupling took {t1 - t0:.4f} seconds")
        # capacity
        constraints.add_constraint_rule(model, name="constraint_capacity_coupling", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"], optimization_setup),
            rule=rules.constraint_capacity_coupling_rule, doc="couples the real capacity variables with the approximated variables")
        t2 = time.perf_counter()
        logging.debug(f"Conversion Technology: constraint_capacity_coupling took {t2 - t1:.4f} seconds")

        # flow coupling constraints for technologies, which are not modeled with an on-off-behavior
        # reference flow coupling
        constraints.add_constraint_block(model, name="constraint_reference_flow_coupling",
                                         constraint=rules.get_constraint_reference_flow_coupling(*cls.create_custom_set(["set_conversion_technologies", "set_no_on_off", "set_dependent_carriers", "set_location", "set_time_steps_operation"], optimization_setup)),
                                         doc="couples the real reference flow variables with the approximated variables")
        t3 = time.perf_counter()
        logging.debug(f"Conversion Technology: constraint_reference_flow_coupling took {t3 - t2:.4f} seconds")
        # dependent flow coupling
        constraints.add_constraint_block(model, name="constraint_dependent_flow_coupling",
                                         constraint=rules.get_constraint_dependent_flow_coupling(*cls.create_custom_set(["set_conversion_technologies", "set_no_on_off", "set_dependent_carriers", "set_location", "set_time_steps_operation"], optimization_setup)),
                                         doc="couples the real dependent flow variables with the approximated variables")
        t4 = time.perf_counter()
        logging.debug(f"Conversion Technology: constraint_dependent_flow_coupling took {t4 - t3:.4f} seconds")

    # defines disjuncts if technology on/off
    @classmethod
    def disjunct_on_technology_rule(cls, optimization_setup, tech, capacity_type, node, time):
        """definition of disjunct constraints if technology is On"""
        # get parameter object
        model = optimization_setup.model
        params = optimization_setup.parameters
        constraints = optimization_setup.constraints
        sets = optimization_setup.sets
        energy_system = optimization_setup.energy_system
        reference_carrier = sets["set_reference_carriers"][tech][0]
        if reference_carrier in sets["set_input_carriers"][tech]:
            reference_flow = model.variables["input_flow"][tech, reference_carrier, node, time]
        else:
            reference_flow = model.variables["output_flow"][tech, reference_carrier, node, time]
        # get invest time step
        time_step_year = energy_system.time_steps.convert_time_step_operation2year(tech, time)
        # disjunct constraints min load
        model.add_constraints(reference_flow.to_linexpr()
                              - model.variables["capacity"][tech, capacity_type, node, time_step_year].to_linexpr(params.min_load.loc[tech, capacity_type, node, time].item())
                              - constraints.M*model.variables["tech_on_var"]
                              >= -constraints.M)
        # couple reference flows
        rules = ConversionTechnologyRules(optimization_setup)
        constraints.add_constraint_block(model, name=f"constraint_reference_flow_coupling_{'_'.join([str(tech), str(node), str(time)])}",
                                         constraint=rules.get_constraint_reference_flow_coupling([(tech, dependent_carrier, node, time) for dependent_carrier in sets["set_dependent_carriers"][tech]], ["set_conversion_technologies", "set_dependent_carriers", "set_nodes", "set_time_steps_operation"]),
                                         doc="couples the real reference flow variables with the approximated variables", disjunction_var=model.variables["tech_on_var"])
        # couple dependent flows
        constraints.add_constraint_block(model, name=f"constraint_dependent_flow_coupling_{'_'.join([str(tech), str(node), str(time)])}",
                                         constraint=rules.get_constraint_dependent_flow_coupling([(tech, dependent_carrier, node, time) for dependent_carrier in sets["set_dependent_carriers"][tech]], ["set_conversion_technologies", "set_dependent_carriers", "set_nodes", "set_time_steps_operation"]),
                                         doc="couples the real dependent flow variables with the approximated variables", disjunction_var=model.variables["tech_on_var"])

    @classmethod
    def disjunct_off_technology_rule(cls, optimization_setup, tech, capacity_type, node, time):
        """definition of disjunct constraints if technology is off"""
        sets = optimization_setup.sets
        model = optimization_setup.model
        constraints = optimization_setup.constraints
        lhs = sum(model.variables["input_flow"][tech, input_carrier, node, time] for input_carrier in sets["set_input_carriers"][tech]) \
              + sum(model.variables["input_flow"][tech, input_carrier, node, time] for input_carrier in sets["set_input_carriers"][tech]) \
              + sum(model.variables["output_flow"][tech, output_carrier, node, time] for output_carrier in sets["set_output_carriers"][tech])
        # equal cons -> to cons
        model.add_constraints(lhs.to_linexpr() + model.variables["tech_off_var"] * constraints.M <= constraints.M)
        model.add_constraints(lhs.to_linexpr() - model.variables["tech_off_var"] * constraints.M >= -constraints.M)

    @classmethod
    def calculate_pwa_breakpoints_values(cls, optimization_setup, setPWA, type_pwa):
        """ calculates the breakpoints and function values for piecewise affine constraint
        :param optimization_setup: The OptimizationSetup the element is part of
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
                pwa_parameter = optimization_setup.get_attribute_of_specific_element(cls, tech, f"pwa_{type_pwa}")
                pwa_breakpoints[index] = pwa_parameter["capacity"]
                pwa_values[index] = pwa_parameter["capex"]
            elif type_pwa == "conver_efficiency":
                # retrieve pwa variables
                pwa_parameter = optimization_setup.get_attribute_of_specific_element(cls, tech, f"pwa_{type_pwa}")
                pwa_breakpoints[index] = pwa_parameter[optimization_setup.get_attribute_of_all_elements(cls, "reference_carrier")[tech][0]]
                pwa_values[index] = pwa_parameter[index[1]]

        return pwa_breakpoints, pwa_values


class ConversionTechnologyRules:
    """
    Rules for the ConversionTechnology class
    """

    def __init__(self, optimization_setup):
        """
        Inits the rules for a given EnergySystem
        :param optimization_setup: The OptimizationSetup the element is part of
        """

        self.optimization_setup = optimization_setup
        self.energy_system = optimization_setup.energy_system

    ### --- functions with constraint rules --- ###
    def constraint_linear_capex_rule(self, tech, node, time):
        """ if capacity and capex have a linear relationship"""
        # get parameter object
        params = self.optimization_setup.parameters
        model = self.optimization_setup.model
        return (model.variables["capex_approximation"][tech, node, time]
                - params.capex_specific_conversion.loc[tech, node, time].item() * model.variables["capacity_approximation"][tech, node, time]
                == 0)

    def get_constraint_linear_conver_efficiency(self, index_values, index_names):
        """ if reference carrier and dependent carrier have a linear relationship"""
        # get parameter object
        params = self.optimization_setup.parameters
        model = self.optimization_setup.model

        # get all the constraints
        constraints = []
        index = ZenIndex(index_values, index_names)
        for tech, dependent_carrier, node in index.get_unique([0, 1, 2]):
            # get invest time step
            coords = [model.variables.coords["set_time_steps_operation"]]
            times = index.get_values([tech, dependent_carrier, node], 3, dtype=list)
            time_step_year = self.energy_system.time_steps.convert_time_step_operation2year(tech, times).values
            tuples = [(1.0, model.variables["dependent_flow_approximation"].loc[tech, dependent_carrier, node, times]),
                      (-params.conver_efficiency_specific.loc[tech, dependent_carrier, node, time_step_year], model.variables["reference_flow_approximation"].loc[tech, dependent_carrier, node, times])]
            constraints.append(linexpr_from_tuple_np(tuples, coords=coords, model=model)
                               == 0)
        return self.optimization_setup.constraints.combine_constraints(constraints, "constraint_linear_conver_efficiency_dim", model)

    def constraint_capex_coupling_rule(self, tech, node, time):
        """ couples capex variables based on modeling technique"""
        model = self.optimization_setup.model
        return (model.variables["capex"][tech, "power", node, time]
                - model.variables["capex_approximation"][tech, node, time]
                == 0)

    def constraint_capacity_coupling_rule(self, tech, node, time):
        """ couples capacity variables based on modeling technique"""
        model = self.optimization_setup.model
        return (model.variables["built_capacity"][tech, "power", node, time]
                - model.variables["capacity_approximation"][tech, node, time]
                == 0)

    def get_constraint_reference_flow_coupling(self, index_values, index_names):
        """ couples reference flow variables based on modeling technique"""
        model = self.optimization_setup.model
        sets = self.optimization_setup.sets

        # check if we even have something
        if len(index_values) == 0:
            return []

        # get all the constraints
        constraints = []
        index = ZenIndex(index_values, index_names)
        for tech, dependent_carrier in index.get_unique([0, 1]):
            reference_carrier = sets["set_reference_carriers"][tech][0]
            if reference_carrier in sets["set_input_carriers"][tech]:
                constraints.append(model.variables["input_flow"].loc[tech, reference_carrier]
                                   - model.variables["reference_flow_approximation"].loc[tech, dependent_carrier]
                                   == 0)
            else:
                constraints.append(model.variables["output_flow"].loc[tech, reference_carrier]
                                   - model.variables["reference_flow_approximation"].loc[tech, dependent_carrier]
                                   == 0)
        return self.optimization_setup.constraints.combine_constraints(constraints, "constraint_reference_flow_coupling_dim", model)

    def get_constraint_dependent_flow_coupling(self, index_values, index_names):
        """ couples dependent flow variables based on modeling technique"""
        model = self.optimization_setup.model
        sets = self.optimization_setup.sets

        # check if we even have something
        if len(index_values) == 0:
            return []

        # get all the constraints
        constraints = []
        index = ZenIndex(index_values, index_names)
        for tech, dependent_carrier in index.get_unique([0, 1]):
            if dependent_carrier in sets["set_input_carriers"][tech]:
                constraints.append(model.variables["input_flow"].loc[tech, dependent_carrier]
                                   - model.variables["dependent_flow_approximation"].loc[tech, dependent_carrier]
                                   == 0)
            else:
                constraints.append(model.variables["output_flow"].loc[tech, dependent_carrier]
                                   - model.variables["dependent_flow_approximation"].loc[tech, dependent_carrier]
                                   == 0)

        return self.optimization_setup.constraints.combine_constraints(constraints, "constraint_dependent_flow_coupling_dim", model)
