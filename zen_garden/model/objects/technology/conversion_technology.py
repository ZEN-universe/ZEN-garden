"""
:Title: ZEN-GARDEN
:Created: October-2021
:Authors:   Alissa Ganter (aganter@ethz.ch), Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Class defining the parameters, variables, and constraints of the conversion technologies.
The class takes the abstract optimization model as an input and adds parameters, variables, and
constraints of the conversion technologies.
"""
import logging

import numpy as np
import pandas as pd
import xarray as xr

from zen_garden.utils import linexpr_from_tuple_np, InputDataChecks
from .technology import Technology
from ..component import ZenIndex
from ..element import GenericRule


class ConversionTechnology(Technology):
    """
    Class defining conversion technologies
    """
    # set label
    label = "set_conversion_technologies"
    location_type = "set_nodes"

    def __init__(self, tech, optimization_setup):
        """
        init conversion technology object

        :param tech: name of added technology
        :param optimization_setup: The OptimizationSetup the element is part of
        """

        super().__init__(tech, optimization_setup)
        # store carriers of conversion technology
        self.store_carriers()

    def store_carriers(self):
        """ retrieves and stores information on reference, input and output carriers """
        # get reference carrier from class <Technology>
        super().store_carriers()
        # define input and output carrier
        self.input_carrier = self.data_input.extract_carriers(carrier_type="input_carrier")
        self.output_carrier = self.data_input.extract_carriers(carrier_type="output_carrier")
        self.energy_system.set_technology_of_carrier(self.name, self.input_carrier + self.output_carrier)
        # check if reference carrier in input and output carriers and set technology to correspondent carrier
        self.optimization_setup.input_data_checks.check_carrier_configuration(input_carrier=self.input_carrier,
                                                                              output_carrier=self.output_carrier,
                                                                              reference_carrier=self.reference_carrier,
                                                                              name=self.name)

    def store_input_data(self):
        """ retrieves and stores input data for element as attributes. Each Child class overwrites method to store different attributes """
        # get attributes from class <Technology>
        super().store_input_data()
        # get conversion efficiency and capex
        self.get_conversion_factor()
        self.opex_specific_fixed = self.data_input.extract_input_data("opex_specific_fixed", index_sets=["set_nodes", "set_time_steps_yearly"], time_steps="set_time_steps_yearly", unit_category={"money": 1, "energy_quantity": -1, "time": 1})
        self.convert_to_fraction_of_capex()

    def get_conversion_factor(self):
        """retrieves and stores conversion_factor """
        # df_input_linear, has_unit_linear = self.data_input.read_pwa_files("conversion_factor")
        dependent_carrier = list(set(self.input_carrier + self.output_carrier).difference(
                self.reference_carrier))
        if not dependent_carrier:
            self.raw_time_series["conversion_factor"] = None
        else:
            index_sets = ["set_nodes", "set_time_steps"]
            time_steps = "set_base_time_steps_yearly"
            cf_dict = {}
            for carrier in dependent_carrier:
                cf_dict[carrier] = self.data_input.extract_input_data("conversion_factor", index_sets=index_sets, unit_category=None, time_steps=time_steps, subelement=carrier)
            cf_dict = pd.DataFrame.from_dict(cf_dict)
            cf_dict.columns.name = "carrier"
            cf_dict = cf_dict.stack()
            conversion_factor_levels = [cf_dict.index.names[-1]] + cf_dict.index.names[:-1]
            cf_dict = cf_dict.reorder_levels(conversion_factor_levels)
            # extract yearly variation
            self.data_input.extract_yearly_variation("conversion_factor", index_sets)
            self.raw_time_series["conversion_factor"] = cf_dict

    def convert_to_fraction_of_capex(self):
        """ this method retrieves the total capex and converts it to annualized capex """
        pwa_capex, self.capex_is_pwa = self.data_input.extract_pwa_capex()
        # annualize cost_capex
        fraction_year = self.calculate_fraction_of_year()
        self.opex_specific_fixed = self.opex_specific_fixed * fraction_year
        if not self.capex_is_pwa:
            self.capex_specific = pwa_capex["capex"] * fraction_year
        else:
            self.pwa_capex = pwa_capex
            self.pwa_capex["capex"] = [value * fraction_year for value in self.pwa_capex["capex"]]
            # set bounds
            self.pwa_capex["bounds"]["capex"] = tuple([(bound * fraction_year) for bound in self.pwa_capex["bounds"]["capex"]])
        # calculate capex of existing capacity
        self.capex_capacity_existing = self.calculate_capex_of_capacities_existing()

    def calculate_capex_of_single_capacity(self, capacity, index):
        """ this method calculates the annualized capex of a single existing capacity.

        :param capacity: existing capacity of technology
        :param index: index of capacity specifying node and time
        :return: annualized capex of a single existing capacity
        """
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
    def get_capex_all_elements(cls, optimization_setup, index_names=None):
        """ similar to Element.get_attribute_of_all_elements but only for capex.
        If select_pwa, extract pwa attributes, otherwise linear.

        :param optimization_setup: The OptimizationSetup the element is part of
        :param select_pwa: boolean if get attributes for pwa
        :return dict_of_attributes: returns dict of attribute values """
        class_elements = optimization_setup.get_all_elements(cls)
        dict_of_attributes = {}
        dict_of_units = {}
        is_pwa_attribute = "capex_is_pwa"
        attribute_name_linear = "capex_specific"

        for element in class_elements:
            # extract for pwa
            if not getattr(element, is_pwa_attribute):
                dict_of_attributes, _, dict_of_units = optimization_setup.append_attribute_of_element_to_dict(element, attribute_name_linear, dict_of_attributes, dict_of_units={})
        if not dict_of_attributes:
            _, index_names = cls.create_custom_set(index_names, optimization_setup)
            return dict_of_attributes, index_names, dict_of_units
        dict_of_attributes = pd.concat(dict_of_attributes, keys=dict_of_attributes.keys())
        if not index_names:
            logging.warning(f"Initializing the parameter capex without the specifying the index names will be deprecated!")
            return dict_of_attributes, dict_of_units
        else:
            custom_set, index_names = cls.create_custom_set(index_names, optimization_setup)
            dict_of_attributes = optimization_setup.check_for_subindex(dict_of_attributes, custom_set)
            return dict_of_attributes, index_names, dict_of_units

    ### --- classmethods to construct sets, parameters, variables, and constraints, that correspond to ConversionTechnology --- ###
    @classmethod
    def construct_sets(cls, optimization_setup):
        """ constructs the pe.Sets of the class <ConversionTechnology>

        :param optimization_setup: The OptimizationSetup the element is part of """
        model = optimization_setup.model
        # get input carriers
        input_carriers = optimization_setup.get_attribute_of_all_elements(cls, "input_carrier")
        output_carriers = optimization_setup.get_attribute_of_all_elements(cls, "output_carrier")
        reference_carrier = optimization_setup.get_attribute_of_all_elements(cls, "reference_carrier")
        dependent_carriers = {}
        for tech in input_carriers:
            dependent_carriers[tech] = input_carriers[tech] + output_carriers[tech]
            dependent_carriers[tech].remove(reference_carrier[tech][0])
        # input carriers of technology
        optimization_setup.sets.add_set(name="set_input_carriers", data=input_carriers,
                                        doc="set of carriers that are an input to a specific conversion technology. Indexed by set_conversion_technologies",
                                        index_set="set_conversion_technologies")
        # output carriers of technology
        optimization_setup.sets.add_set(name="set_output_carriers", data=output_carriers,
                                        doc="set of carriers that are an output to a specific conversion technology. Indexed by set_conversion_technologies",
                                        index_set="set_conversion_technologies")
        # dependent carriers of technology
        optimization_setup.sets.add_set(name="set_dependent_carriers", data=dependent_carriers,
                                        doc="set of carriers that are an output to a specific conversion technology. Indexed by set_conversion_technologies",
                                        index_set="set_conversion_technologies")

        # add sets of the child classes
        for subclass in cls.__subclasses__():
            if np.size(optimization_setup.system[subclass.label]):
                subclass.construct_sets(optimization_setup)

    @classmethod
    def construct_params(cls, optimization_setup):
        """ constructs the pe.Params of the class <ConversionTechnology>

        :param optimization_setup: The OptimizationSetup the element is part of """
        # slope of linearly modeled capex
        optimization_setup.parameters.add_parameter(name="capex_specific_conversion", index_names=["set_conversion_technologies", "set_capex_linear", "set_nodes", "set_time_steps_yearly"],
            doc="Parameter which specifies the slope of the capex if approximated linearly", calling_class=cls)
        # slope of linearly modeled conversion efficiencies
        optimization_setup.parameters.add_parameter(name="conversion_factor", index_names=["set_conversion_technologies", "set_dependent_carriers", "set_nodes", "set_time_steps_operation"],
            doc="Parameter which specifies the conversion factor", calling_class=cls)

        # add params of the child classes
        for subclass in cls.__subclasses__():
            if np.size(optimization_setup.system[subclass.label]):
                subclass.construct_params(optimization_setup)

    @classmethod
    def construct_vars(cls, optimization_setup):
        """ constructs the pe.Vars of the class <ConversionTechnology>

        :param optimization_setup: The OptimizationSetup the element is part of """

        model = optimization_setup.model
        variables = optimization_setup.variables

        def flow_conversion_bounds(index_values, index_names):
            """ return bounds of carrier_flow for bigM expression
            :param index_values: list of index values
            :return bounds: bounds of carrier_flow"""
            params = optimization_setup.parameters
            sets = optimization_setup.sets
            energy_system = optimization_setup.energy_system

            # init the bounds
            index_arrs = sets.tuple_to_arr(index_values, index_names)
            coords = [optimization_setup.sets.get_coord(data, name) for data, name in zip(index_arrs, index_names)]
            lower = xr.DataArray(0.0, coords=coords)
            upper = xr.DataArray(np.inf, coords=coords)

            # get the sets
            technology_set, carrier_set, node_set, timestep_set = [sets[name] for name in index_names]

            for tech in technology_set:
                for carrier in carrier_set[tech]:
                    time_step_year = [energy_system.time_steps.convert_time_step_operation2year(t) for t in timestep_set]
                    if carrier == sets["set_reference_carriers"][tech][0]:
                        conversion_factor_lower = 1
                        conversion_factor_upper = 1
                    else:
                        conversion_factor_lower = params.conversion_factor.loc[tech, carrier, node_set].min().data
                        conversion_factor_upper = params.conversion_factor.loc[tech, carrier, node_set].max().data
                        if 0 in conversion_factor_upper:
                            _rounding_ts = optimization_setup.solver.rounding_decimal_points_ts
                            raise ValueError(f"Maximum conversion factor of {tech} for carrier {carrier} is 0.\nOne reason might be that the conversion factor is too small (1e-{_rounding_ts}), so that it is rounded to 0 after the time series aggregation.")

                    lower.loc[tech, carrier, ...] = model.variables["capacity"].lower.loc[tech, "power", node_set, time_step_year].data * conversion_factor_lower
                    upper.loc[tech, carrier, ...] = model.variables["capacity"].upper.loc[tech, "power", node_set, time_step_year].data * conversion_factor_upper

            # make sure lower is never below 0
            return (lower, upper)

        ## Flow variables
        # input flow of carrier into technology
        index_values, index_names = cls.create_custom_set(["set_conversion_technologies", "set_input_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup)
        variables.add_variable(model, name="flow_conversion_input", index_sets=(index_values, index_names),
            bounds=flow_conversion_bounds(index_values, index_names), doc='Carrier input of conversion technologies', unit_category={"energy_quantity": 1, "time": -1})
        # output flow of carrier into technology
        index_values, index_names = cls.create_custom_set(["set_conversion_technologies", "set_output_carriers", "set_nodes", "set_time_steps_operation"], optimization_setup)
        variables.add_variable(model, name="flow_conversion_output", index_sets=(index_values, index_names),
            bounds=flow_conversion_bounds(index_values, index_names), doc='Carrier output of conversion technologies', unit_category={"energy_quantity": 1, "time": -1})
        ## pwa Variables - Capex
        # pwa capacity
        variables.add_variable(model, name="capacity_approximation", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"], optimization_setup), bounds=(0,np.inf),
            doc='pwa variable for size of installed technology on edge i and time t', unit_category={"energy_quantity": 1, "time": -1})
        # pwa capex technology
        variables.add_variable(model, name="capex_approximation", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"], optimization_setup), bounds=(0,np.inf),
            doc='pwa variable for capex for installing technology on edge i and time t', unit_category={"money": 1})

    @classmethod
    def construct_constraints(cls, optimization_setup):
        """ constructs the pe.Constraints of the class <ConversionTechnology>

        :param optimization_setup: optimization setup"""
        model = optimization_setup.model
        constraints = optimization_setup.constraints
        # add pwa constraints
        rules = ConversionTechnologyRules(optimization_setup)
        # conversion factor
        constraints.add_constraint_block(model, name="constraint_carrier_conversion",
                                         constraint=rules.constraint_carrier_conversion_block(*cls.create_custom_set(["set_conversion_technologies", "set_dependent_carriers", "set_nodes","set_time_steps_operation"], optimization_setup)),
                                         doc="Conversion of energy carrier with conversion_factor")
        # capex
        set_pwa_capex = cls.create_custom_set(["set_conversion_technologies", "set_capex_pwa", "set_nodes", "set_time_steps_yearly"], optimization_setup)
        set_linear_capex = cls.create_custom_set(["set_conversion_technologies", "set_capex_linear", "set_nodes", "set_time_steps_yearly"], optimization_setup)
        if len(set_pwa_capex[0]) > 0:
            # if set_pwa_capex contains technologies:
            pwa_breakpoints, pwa_values = cls.calculate_capex_pwa_breakpoints_values(optimization_setup, set_pwa_capex[0])
            constraints.add_pw_constraint(model, index_values=set_pwa_capex[0], yvar="capex_approximation", xvar="capacity_approximation",
                                          break_points=pwa_breakpoints, f_vals=pwa_values, cons_type="EQ", name="constraint_capex_pwa",)
        if set_linear_capex[0]:
            # if set_linear_capex contains technologies: (note we give the coordinates nice names)
            constraints.add_constraint_rule(model, name="constraint_linear_capex",
                                            index_sets=(set_linear_capex[0], ["lin_capex_tech", "lin_capex_node", "lin_capex_time_step"]),
                                            rule=rules.constraint_linear_capex_rule, doc="Linear relationship in capex")
        # Coupling constraints
        # couple the real variables with the auxiliary variables
        constraints.add_constraint_rule(model, name="constraint_capex_coupling", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"], optimization_setup),
            rule=rules.constraint_capex_coupling_rule, doc="couples the real capex variables with the approximated variables")
        # capacity
        constraints.add_constraint_rule(model, name="constraint_capacity_coupling", index_sets=cls.create_custom_set(["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"], optimization_setup),
            rule=rules.constraint_capacity_coupling_rule, doc="couples the real capacity variables with the approximated variables")


        # add constraints of the child classes
        for subclass in cls.__subclasses__():
           if np.size(optimization_setup.system[subclass.label]):
                subclass.construct_constraints(optimization_setup)

    # defines disjuncts if technology on/off
    @classmethod
    def disjunct_on_technology_rule(cls, optimization_setup, tech, capacity_type, node, time, binary_var):
        """definition of disjunct constraints if technology is On

        :param optimization_setup: optimization setup
        :param tech: technology
        :param capacity_type: type of capacity (power, energy)
        :param node: node
        :param time: yearly time step
        :param binary_var: binary disjunction variable
        """
        # get parameter object
        model = optimization_setup.model
        params = optimization_setup.parameters
        constraints = optimization_setup.constraints
        sets = optimization_setup.sets
        energy_system = optimization_setup.energy_system
        reference_carrier = sets["set_reference_carriers"][tech][0]
        if reference_carrier in sets["set_input_carriers"][tech]:
            reference_flow = model.variables["flow_conversion_input"].loc[tech, reference_carrier, node, time]
        else:
            reference_flow = model.variables["flow_conversion_output"].loc[tech, reference_carrier, node, time]
        # get invest time step
        time_step_year = energy_system.time_steps.convert_time_step_operation2year(time)
        # formulate constraint
        lhs = reference_flow - params.min_load.loc[tech, capacity_type, node, time]* model.variables["capacity"].loc[tech, capacity_type, node, time_step_year]
        rhs = 0
        constraint = lhs >= rhs
        # disjunct constraints min load
        constraints.add_constraint_block(model, name=f"disjunct_conversion_technology_min_load_{tech}_{capacity_type}_{node}_{time}",
                                         constraint=constraint, disjunction_var=binary_var)

    @classmethod
    def disjunct_off_technology_rule(cls, optimization_setup, tech, capacity_type, node, time, binary_var):
        """definition of disjunct constraints if technology is off

        :param optimization_setup: optimization setup
        :param tech: technology
        :param capacity_type: type of capacity (power, energy)
        :param node: node
        :param time: yearly time step
        :param binary_var: binary disjunction variable
        """
        sets = optimization_setup.sets
        model = optimization_setup.model
        constraints = optimization_setup.constraints
        lhs = sum(model.variables["flow_conversion_input"].loc[tech, input_carrier, node, time] for input_carrier in sets["set_input_carriers"][tech]) \
              + sum(model.variables["flow_conversion_output"].loc[tech, output_carrier, node, time] for output_carrier in sets["set_output_carriers"][tech])
        # add the constraints
        constraints.add_constraint_block(model, name=f"disjunct_conversion_technology_off_{tech}_{capacity_type}_{node}_{time}",
                                         constraint=lhs == 0, disjunction_var=binary_var)

    @classmethod
    def calculate_capex_pwa_breakpoints_values(cls, optimization_setup, set_pwa):
        """ calculates the breakpoints and function values for piecewise affine constraint

        :param optimization_setup: The OptimizationSetup the element is part of
        :param set_pwa: set of variable indices in capex approximation, for which pwa is performed
        :return pwa_breakpoints: dict of pwa breakpoint values
        :return pwa_values: dict of pwa function values """
        pwa_breakpoints = {}
        pwa_values = {}

        # iterate through pwa variable's indices
        for index in set_pwa:
            pwa_breakpoints[index] = []
            pwa_values[index] = []
            if len(index) > 1:
                tech = index[0]
            else:
                tech = index
            # retrieve pwa variables
            pwa_parameter = optimization_setup.get_attribute_of_specific_element(cls, tech, f"pwa_capex")
            pwa_breakpoints[index] = pwa_parameter["capacity"]
            pwa_values[index] = pwa_parameter["capex"]
        return pwa_breakpoints, pwa_values

    @classmethod
    def get_flow_term_reference_carrier(cls, optimization_setup, tech):
        """ get reference carrier flow term of conversion technology

        :param optimization_setup: The OptimizationSetup the element is part of
        :param tech: conversion technology
        :return term_flow: return reference carrier flow term """
        model = optimization_setup.model
        sets = optimization_setup.sets
        reference_carrier = sets["set_reference_carriers"][tech][0]
        if reference_carrier in sets["set_input_carriers"][tech]:
            term_flow = model.variables["flow_conversion_input"].loc[tech, reference_carrier]
        else:
            term_flow = model.variables["flow_conversion_output"].loc[tech, reference_carrier]
        return term_flow


class ConversionTechnologyRules(GenericRule):
    """
    Rules for the ConversionTechnology class
    """

    def __init__(self, optimization_setup):
        """
        Inits the rules for a given EnergySystem
        :param optimization_setup: The OptimizationSetup the element is part of
        """

        super().__init__(optimization_setup)


    # Rule-based constraints
    # -----------------------

    def constraint_linear_capex_rule(self, tech, node, year):
        """ if capacity and capex have a linear relationship

        .. math::
            A_{h,p,y}^{approximation} = \\alpha_{h,n,y} S_{h,p,y}^{approximation}

        :param tech: technology
        :param node: node
        :param year: yearly time step
        :return: linopy constraints
        """

        ### index sets
        # skipped because rule-based constraint

        ### masks
        # skipped because rule-based constraint

        ### index loop
        # skipped because rule-based constraint

        ### auxiliary calculations
        # not necessary

        ### formulate constraint
        lhs = (self.variables["capex_approximation"][tech, node, year]
               - self.parameters.capex_specific_conversion.loc[tech, node, year].item() * self.variables["capacity_approximation"][tech, node, year])
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)

    def constraint_capex_coupling_rule(self, tech, node, year):
        """ couples capex variables based on modeling technique

        .. math::
            CAPEX_{y,n,i}^\\mathrm{cost, power} = A_{h,p,y}^{approximation}

        :param tech: technology
        :param node: node
        :param year: yearly time step
        :return: linopy constraints
        """

        ### index sets
        # skipped because rule-based constraint

        ### masks
        # skipped because rule-based constraint

        ### index loop
        # skipped because rule-based constraint

        ### auxiliary calculations
        # not necessary

        ### formulate constraint
        lhs = (self.variables["cost_capex"][tech, "power", node, year]
               - self.variables["capex_approximation"][tech, node, year])
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)

    def constraint_capacity_coupling_rule(self, tech, node, year):
        """ couples capacity variables based on modeling technique

        .. math::
            \Delta S_{h,p,y}^\mathrm{power} = S_{h,p,y}^\mathrm{approximation}

        :param tech: technology
        :param node: node
        :param year: yearly time step
        :return: linopy constraints
        """

        ### index sets
        # skipped because rule-based constraint

        ### masks
        # skipped because rule-based constraint

        ### index loop
        # skipped because rule-based constraint

        ### auxiliary calculations
        # not necessary

        ### formulate constraint
        lhs = (self.variables["capacity_addition"][tech, "power", node, year]
               - self.variables["capacity_approximation"][tech, node, year])
        rhs = 0
        constraints = lhs == rhs

        ### return
        return self.constraints.return_contraints(constraints)

    # Block-based constraints
    # -----------------------

    def constraint_carrier_conversion_block(self, index_values, index_names):
        """ conversion factor between reference carrier and dependent carrier

        .. math::
            G^\\mathrm{d}_{i,n,t} = \\eta_{i,c,n,y}G^\\mathrm{r}_{i,n,t}

        :param index_values: index values
        :param index_names: index names
        :return: linopy constraints
        """

        ### index sets
        index = ZenIndex(index_values, index_names)

        ### masks
        # note necessary

        ### index loop
        # we loop over all technologies and dependent carriers, mostly to avoid renaming of dimension
        constraints = []
        for tech, dependent_carrier in index.get_unique(["set_conversion_technologies", "set_dependent_carriers"]):

            ### auxiliary calculations
            reference_carrier = self.sets["set_reference_carriers"][tech][0]
            if reference_carrier in self.sets["set_input_carriers"][tech]:
                term_flow_reference = self.variables["flow_conversion_input"].loc[tech, reference_carrier]
            else:
                term_flow_reference = self.variables["flow_conversion_output"].loc[tech, reference_carrier]
            if dependent_carrier in self.sets["set_input_carriers"][tech]:
                term_flow_dependent = self.variables["flow_conversion_input"].loc[tech, dependent_carrier]
            else:
                term_flow_dependent = self.variables["flow_conversion_output"].loc[tech, dependent_carrier]
            ### formulate constraint
            lhs = term_flow_dependent - self.parameters.conversion_factor.loc[tech, dependent_carrier]*term_flow_reference
            rhs = 0
            constraints.append(lhs == rhs)

        ### return
        return self.constraints.return_contraints(constraints,model=self.model,
                                                  index_values=index.get_unique(["set_conversion_technologies", "set_dependent_carriers"]),
                                                  index_names=["set_conversion_technologies", "set_dependent_carriers"])
