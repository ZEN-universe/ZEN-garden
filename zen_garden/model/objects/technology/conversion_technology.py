"""
:Title: ZEN-GARDEN
:Created: October-2021
:Authors:   Alissa Ganter (aganter@ethz.ch), Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Class defining the parameters, variables, and constraints of the conversion technologies.
The class takes the abstract optimization model as an input and adds parameters, variables, and
constraints of the conversion technologies.
"""
import itertools
import logging

import numpy as np
import pandas as pd
import xarray as xr
import linopy as lp
from zen_garden.utils import align_like
from .technology import Technology
from ..component import ZenIndex
from ..element import GenericRule,Element


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
            self.capex_specific_conversion = pwa_capex["capex"] * fraction_year
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
            capex = self.capex_specific_conversion[index[0]].iloc[0] * capacity
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
        attribute_name_linear = "capex_specific_conversion"

        for element in class_elements:
            # extract for pwa
            if not getattr(element, is_pwa_attribute):
                dict_of_attributes, _, dict_of_units = optimization_setup.append_attribute_of_element_to_dict(element, attribute_name_linear, dict_of_attributes, dict_of_units=dict_of_units)
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
                            _rounding_tsa = optimization_setup.solver.rounding_decimal_points_tsa
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
        """ constructs the Constraints of the class <ConversionTechnology>

        :param optimization_setup: optimization setup"""
        model = optimization_setup.model
        constraints = optimization_setup.constraints
        # add pwa constraints
        rules = ConversionTechnologyRules(optimization_setup)
        # capacity factor constraint
        rules.constraint_capacity_factor_conversion()
        # opex and emissions constraint for conversion technologies
        rules.constraint_opex_emissions_technology_conversion()
        # conversion factor
        rules.constraint_carrier_conversion()

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
            rules.constraint_linear_capex()
        # Coupling constraints
        rules.constraint_capacity_capex_coupling()

        # add constraints of the child classes
        for subclass in cls.__subclasses__():
           if np.size(optimization_setup.system[subclass.label]):
                subclass.construct_constraints(optimization_setup)

    # defines disjuncts if technology on/off
    @classmethod
    def disjunct_on_technology(cls, optimization_setup, tech, capacity_type, node, time, binary_var):
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
        # TODO make to constraint rule or integrate in new structure!!!
        constraints.add_constraint_block(model, name=f"disjunct_conversion_technology_min_load_{tech}_{capacity_type}_{node}_{time}",
                                         constraint=constraint, disjunction_var=binary_var)

    @classmethod
    def disjunct_off_technology(cls, optimization_setup, tech, capacity_type, node, time, binary_var):
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
            pwa_breakpoints[index] = pwa_parameter["capacity_addition"]
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


    def constraint_capacity_factor_conversion(self):
        """ Load is limited by the installed capacity and the maximum load factor

        .. math::
            G_{i,n,t,y}^\mathrm{r} \\leq m_{i,n,t,y}S_{i,n,y}

        """
        techs = self.sets["set_conversion_technologies"]
        if len(techs) == 0:
            return
        nodes = self.sets["set_nodes"]
        times = self.parameters.max_load.coords["set_time_steps_operation"]
        time_step_year = xr.DataArray([self.optimization_setup.energy_system.time_steps.convert_time_step_operation2year(t) for t in times.data], coords=[times])
        term_capacity = (
                self.parameters.max_load.loc[techs, "power", nodes, :]
                * self.variables["capacity"].loc[techs, "power", nodes, time_step_year]
            ).rename({"set_technologies": "set_conversion_technologies","set_location": "set_nodes"})
        term_reference_flow = self.get_flow_expression_conversion(techs,nodes)
        lhs = term_capacity + term_reference_flow
        rhs = 0
        constraints = lhs >= rhs

        self.constraints.add_constraint("constraint_capacity_factor_conversion",constraints)

    def constraint_opex_emissions_technology_conversion(self):
        """ calculate opex and carbon emissions of each technology

        .. math::
            OPEX_{h,p,t}^\mathrm{cost} = \\beta_{h,p,t} G_{i,n,t,y}^\mathrm{r}
            E_{h,p,t} = \\epsilon_h G_{i,n,t,y}^\mathrm{r}

        """
        techs = self.sets["set_conversion_technologies"]
        if len(techs) == 0:
            return
        nodes = self.sets["set_nodes"]
        term_reference_flow_opex = self.get_flow_expression_conversion(techs,nodes,factor=self.parameters.opex_specific_variable.rename({"set_technologies": "set_conversion_technologies","set_location":"set_nodes"}))
        term_reference_flow_emissions = self.get_flow_expression_conversion(techs,nodes,factor=self.parameters.carbon_intensity_technology.rename({"set_technologies": "set_conversion_technologies","set_location":"set_nodes"}))
        lhs_opex = ((1*self.variables["cost_opex"].loc[techs,nodes,:]).rename({"set_technologies": "set_conversion_technologies","set_location":"set_nodes"}) + term_reference_flow_opex)
        lhs_emissions = ((1*self.variables["carbon_emissions_technology"].loc[techs,nodes,:]).rename({"set_technologies": "set_conversion_technologies","set_location":"set_nodes"}) + term_reference_flow_emissions)
        rhs = 0
        constraints_opex = lhs_opex == rhs
        constraints_emissions = lhs_emissions == rhs

        self.constraints.add_constraint("constraint_opex_technology_conversion",constraints_opex)
        self.constraints.add_constraint("constraint_carbon_emissions_technology_conversion",constraints_emissions)

    def constraint_linear_capex(self):
        """ if capacity and capex have a linear relationship

        .. math::
            A_{h,p,y}^{approximation} = \\alpha_{h,n,y} S_{h,p,y}^{approximation}

        """
        capex_specific_conversion = self.parameters.capex_specific_conversion
        capex_specific_conversion = capex_specific_conversion.rename({old: new for old, new in zip(list(capex_specific_conversion.dims),
                                          ["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"])})
        capex_specific_conversion = capex_specific_conversion.broadcast_like(self.variables["capacity_approximation"].lower)
        mask = ~np.isnan(capex_specific_conversion)
        lhs = lp.merge(
            [1 * self.variables["capex_approximation"],
             - capex_specific_conversion * self.variables["capacity_approximation"]],
            compat="broadcast_equals")
        lhs = self.align_and_mask(lhs, mask)
        rhs = 0
        constraints = lhs == rhs

        self.constraints.add_constraint("constraint_linear_capex",constraints)

    def constraint_capacity_capex_coupling(self):
        """ couples capacity variables based on modeling technique

        .. math::
            \Delta S_{h,p,y}^\mathrm{power} = S_{h,p,y}^\mathrm{approximation}

        """

        techs = self.sets["set_conversion_technologies"]
        nodes = self.sets["set_nodes"]
        capacity_addition = self.variables["capacity_addition"].loc[techs,"power",nodes].rename(
            {"set_technologies": "set_conversion_technologies", "set_location": "set_nodes"})
        cost_capex = self.variables["cost_capex"].loc[techs,"power",nodes].rename(
            {"set_technologies": "set_conversion_technologies", "set_location": "set_nodes"})

        ### formulate constraint
        lhs_capacity = capacity_addition - self.variables["capacity_approximation"]
        lhs_capex = cost_capex - self.variables["capex_approximation"]
        rhs = 0
        constraints_capacity = lhs_capacity == rhs
        constraints_capex = lhs_capex == rhs
        ### return
        self.constraints.add_constraint("constraint_capacity_coupling",constraints_capacity)
        self.constraints.add_constraint("constraint_capex_coupling", constraints_capex)

    def constraint_carrier_conversion(self):
        """ conversion factor between reference carrier and dependent carrier

        .. math::
            G^\\mathrm{d}_{i,n,t} = \\eta_{i,c,n,y}G^\\mathrm{r}_{i,n,t}

        """
        # dependent carriers
        flow_conversion_input_dep = self.variables["flow_conversion_input"].rename({"set_input_carriers": "set_dependent_carriers"})
        flow_conversion_output_dep = self.variables["flow_conversion_output"].rename({"set_output_carriers": "set_dependent_carriers"})
        dc_in = pd.Series(
            {(t, c): True if c in self.sets["set_dependent_carriers"][t] else False for t, c in
             itertools.product(self.sets["set_conversion_technologies"],
                               self.sets["set_input_carriers"].superset)})
        dc_out = pd.Series(
            {(t, c): True if c in self.sets["set_dependent_carriers"][t] else False for t, c in
             itertools.product(self.sets["set_conversion_technologies"],
                               self.sets["set_output_carriers"].superset)})
        dc_in.index.names = ["set_conversion_technologies", "set_dependent_carriers"]
        dc_out.index.names = ["set_conversion_technologies", "set_dependent_carriers"]
        combined_dependent_index = xr.align(flow_conversion_input_dep.lower, flow_conversion_output_dep.lower, join="outer")[0]
        dc_in = align_like(dc_in.to_xarray(), combined_dependent_index, astype=bool)
        dc_out = align_like(dc_out.to_xarray(), combined_dependent_index, astype=bool)
        dc = dc_in | dc_out
        term_flow_dependent = lp.merge([1 * flow_conversion_input_dep, 1 * flow_conversion_output_dep], compat="broadcast_equals").where(dc)
        conversion_factor = align_like(self.parameters.conversion_factor, term_flow_dependent)
        # reference carriers
        flow_conversion_input = self.variables["flow_conversion_input"].broadcast_like(conversion_factor)
        flow_conversion_output = self.variables["flow_conversion_output"].broadcast_like(conversion_factor)
        rc_in = pd.Series(
            {(t, c): True if c in self.sets["set_reference_carriers"][t] else False for t, c in
             itertools.product(self.sets["set_conversion_technologies"],
                               self.sets["set_input_carriers"].superset)})
        rc_out = pd.Series(
            {(t, c): True if c in self.sets["set_reference_carriers"][t] else False for t, c in
             itertools.product(self.sets["set_conversion_technologies"],
                               self.sets["set_output_carriers"].superset)})
        rc_in.index.names = ["set_conversion_technologies", "set_input_carriers"]
        rc_out.index.names = ["set_conversion_technologies", "set_output_carriers"]
        rc_in = align_like(rc_in.to_xarray(), flow_conversion_input)
        rc_out = align_like(rc_out.to_xarray(), flow_conversion_output)
        term_flow_reference = (
                flow_conversion_input.where(rc_in).sum("set_input_carriers")
                + flow_conversion_output.where(rc_out).sum("set_output_carriers"))
        # formulate constraint
        lhs = term_flow_dependent - conversion_factor * term_flow_reference
        rhs = 0
        constraints = lhs == rhs

        self.constraints.add_constraint("constraint_carrier_conversion",constraints)

    def get_flow_expression_conversion(self,techs,nodes,factor=None, rename =False):
        """ return the flow expression for conversion technologies """
        reference_flows = []
        for t in techs:
            rc = self.sets["set_reference_carriers"][t][0]
            if factor is not None:
                mult = factor.loc[t, nodes]
            else:
                mult = 1
            # TODO can we avoid the indexing here?
            if rc in self.sets["set_input_carriers"][t]:
                reference_flows.append(-mult * self.variables["flow_conversion_input"].loc[t, rc, nodes, :])
            else:
                reference_flows.append(-mult * self.variables["flow_conversion_output"].loc[t, rc, nodes, :])
        if rename:
            term_reference_flow = lp.merge(reference_flows, dim="set_technologies").rename({"set_nodes":"set_location"})
        else:
            term_reference_flow = lp.merge(reference_flows, dim="set_conversion_technologies")
        return term_reference_flow
