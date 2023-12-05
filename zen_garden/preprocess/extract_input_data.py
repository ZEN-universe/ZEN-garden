"""
:Title:         ZEN-GARDEN
:Created:       January-2022
:Authors:       Jacob Mannhardt (jmannhardt@ethz.ch),
                Alissa Ganter (aganter@ethz.ch)
:Organization:  Laboratory of Risk and Reliability Engineering, ETH Zurich

Functions to extract the input data from the provided input files
"""
import copy
import logging
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import linregress

from zen_garden.utils import InputDataChecks


class DataInput:
    """
    Class to extract input data
    """
    def __init__(self, element, system, analysis, solver, energy_system, unit_handling):
        """ data input object to extract input data

        :param element: element for which data is extracted
        :param system: dictionary defining the system
        :param analysis: dictionary defining the analysis framework
        :param solver: dictionary defining the solver 
        :param energy_system: instance of class <EnergySystem> to define energy_system
        :param unit_handling: instance of class <UnitHandling> to convert units """
        self.element = element
        self.system = system
        self.analysis = analysis
        self.solver = solver
        self.energy_system = energy_system
        self.scenario_dict = None
        self.unit_handling = unit_handling
        # extract folder path
        self.folder_path = getattr(self.element, "input_path")

        # get names of indices
        # self.index_names     = {index_name: self.analysis['header_data_inputs'][index_name][0] for index_name in self.analysis['header_data_inputs']}
        self.index_names = self.analysis['header_data_inputs']

    def extract_input_data(self, file_name, index_sets, unit_category, time_steps=None):
        """ reads input data and restructures the dataframe to return (multi)indexed dict

        :param file_name: name of selected file.
        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param time_steps: string specifying time_steps of element
        :return dataDict: dictionary with attribute values """

        # generic time steps
        yearly_variation = False
        if not time_steps:
            time_steps = "set_base_time_steps"
        # if time steps are the yearly base time steps
        elif time_steps == "set_base_time_steps_yearly":
            yearly_variation = True
            self.extract_yearly_variation(file_name, index_sets, unit_category)

        # if existing capacities and existing capacities not used
        if (file_name == "capacity_existing" or file_name == "capacity_existing_energy") and not self.analysis["use_capacities_existing"]:
            df_output, *_ = self.create_default_output(index_sets, unit_category, file_name=file_name, time_steps=time_steps, manual_default_value=0)
            return df_output
        # use distances computed with node coordinates as default values
        elif file_name == "distance":
            df_output, default_value, index_name_list = self.create_default_output(index_sets, unit_category, file_name=file_name, time_steps=time_steps, manual_default_value=self.energy_system.set_haversine_distances_edges)
        else:
            df_output, default_value, index_name_list = self.create_default_output(index_sets, unit_category, file_name=file_name, time_steps=time_steps)
        # read input file
        f_name, scenario_factor = self.scenario_dict.get_param_file(self.element.name, file_name)
        df_input = self.read_input_data(f_name)
        if f_name != file_name and yearly_variation and df_input is None:
            logging.info(f"{f_name} for current scenario is missing from {self.folder_path}. {file_name} is used as input file")
            df_input = self.read_input_data(file_name)

        assert (df_input is not None or default_value is not None), f"input file for attribute {file_name} could not be imported and no default value is given."
        if df_input is not None and not df_input.empty:
            df_output = self.extract_general_input_data(df_input, df_output, file_name, index_name_list, default_value, time_steps)
        # save parameter values for analysis of numerics
        self.save_values_of_attribute(df_output=df_output, file_name=file_name)
        # finally apply the scenario_factor
        return df_output*scenario_factor

    def extract_general_input_data(self, df_input, df_output, file_name, index_name_list, default_value, time_steps):
        """ fills df_output with data from df_input

        :param df_input: raw input dataframe
        :param df_output: empty output dataframe, only filled with default_value
        :param file_name: name of selected file
        :param index_name_list: list of name of indices
        :param default_value: default for dataframe
        :param time_steps: specific time_steps of element
        :return df_output: filled output dataframe """

        df_input = self.convert_real_to_generic_time_indices(df_input, time_steps, file_name, index_name_list)

        assert df_input.columns is not None, f"Input file '{file_name}' has no columns"
        # set index by index_name_list
        missing_index = list(set(index_name_list) - set(index_name_list).intersection(set(df_input.columns)))
        assert len(missing_index) <= 1, f"More than one the requested index sets ({missing_index}) are missing from input file for {file_name}"

        # no indices missing
        if len(missing_index) == 0:
            df_input = DataInput.extract_from_input_without_missing_index(df_input, index_name_list, file_name)
        else:
            missing_index = missing_index[0]
            # check if special case of existing Technology
            if "technology_existing" in missing_index:
                df_output = DataInput.extract_from_input_for_capacities_existing(df_input, df_output, index_name_list, file_name, missing_index)
                if isinstance(default_value, dict):
                    df_output = df_output * default_value["multiplier"]
                return df_output
            # index missing
            else:
                df_input = DataInput.extract_from_input_with_missing_index(df_input, df_output, copy.deepcopy(index_name_list), file_name, missing_index)

        #check for duplicate indices
        df_input = self.energy_system.optimization_setup.input_data_checks.check_duplicate_indices(df_input=df_input, file_name=file_name, folder_path=self.folder_path)

        # apply multiplier to input data
        df_input = df_input * default_value["multiplier"]
        # delete nans
        df_input = df_input.dropna()

        # get common index of df_output and df_input
        if not isinstance(df_input.index, pd.MultiIndex) and isinstance(df_output.index, pd.MultiIndex):
            index_list = df_input.index.to_list()
            if len(index_list) == 1:
                index_multi_index = pd.MultiIndex.from_tuples([(index_list[0],)], names=[df_input.index.name])
            else:
                index_multi_index = pd.MultiIndex.from_product([index_list], names=[df_input.index.name])
            df_input = pd.Series(index=index_multi_index, data=df_input.to_list(),dtype=float)
        common_index = df_output.index.intersection(df_input.index)
        assert default_value is not None or len(common_index) == len(df_output.index), f"Input for {file_name} does not provide entire dataset and no default given in attributes.csv"
        df_output.loc[common_index] = df_input.loc[common_index]
        return df_output

    def read_input_data(self, input_file_name):
        """ reads input data and returns raw input dataframe

        :param input_file_name: name of selected file
        :return df_input: pd.DataFrame with input data """

        # append .csv suffix
        input_file_name += ".csv"

        # select data
        file_names = os.listdir(self.folder_path)
        if input_file_name in file_names:
            df_input = pd.read_csv(os.path.join(self.folder_path, input_file_name), header=0, index_col=None)
            #check for header name duplicates (pd.read_csv() adds a dot and a number to duplicate headers)
            if any("." in col for col in df_input.columns):
                raise AssertionError(f"The input data file {input_file_name} at {self.folder_path} contains two identical header names.")
            return df_input
        else:
            return None

    def extract_attribute(self, attribute_name, unit_category, skip_warning=False, check_if_exists=False):
        """ reads input data and restructures the dataframe to return (multi)indexed dict

        :param attribute_name: name of selected attribute
        :param skip_warning: boolean to indicate if "Default" warning is skipped
        :param check_if_exists: check if attribute exists
        :return attribute_value: attribute value """
        if self.scenario_dict is not None:
            filename, factor = self.scenario_dict.get_default(self.element.name, attribute_name)
        else:
            filename = "attributes"
            factor = 1
        df_input = self.read_input_data(filename)
        if df_input is not None:
            df_input = df_input.set_index("index").squeeze(axis=1)
            attribute_name = self.adapt_attribute_name(attribute_name, df_input, skip_warning, suppress_error=check_if_exists)
        if attribute_name is not None:
            # get attribute
            attribute_value = df_input.loc[attribute_name, "value"]
            attribute_unit = df_input.loc[attribute_name, "unit"]
            multiplier, attribute_unit_in_base_units = self.unit_handling.convert_unit_into_base_units(attribute_unit, get_multiplier=True, attribute_name=attribute_name, path=self.folder_path)
            #don't try to save input-/output carrier if they don't exist for a conversion technology
            if not(pd.isna(attribute_value) and attribute_name in ["input_carrier", "output_carrier"]):
                self.element.units[attribute_name] = unit_category, attribute_unit_in_base_units
            #don't convert unit of conversion factor to base units since e.g. kWh/kWh would become 1 (however, conversion factors' unit consistency must be checked with the corresponding carriers)
            if attribute_name == "conversion_factor_default":
                self.element.units[attribute_name] = unit_category, attribute_unit
            try:
                attribute = {"value": float(attribute_value) * multiplier * factor, "multiplier": multiplier}
                return attribute
            except ValueError:
                if factor != 1:
                    logging.warning(f"WARNING: Attribute {attribute_name} of {self.element.name} is not a number "
                                    f"but has custom factor {factor}, factor will be ignored...")
                return attribute_value
        else:
            return None

    def adapt_attribute_name(self, attribute_name, df_input, skip_warning=False, suppress_error=False):
        """ check if attribute in index

        :param attribute_name: name of selected attribute
        :param df_input: pd.DataFrame with input data
        :param skip_warning: boolean to indicate if "Default" warning is skipped
        :param suppress_error: suppress AttributeError if only check for existence of attribute
        :return:
        """
        if attribute_name + "_default" not in df_input.index:
            if attribute_name not in df_input.index:
                if suppress_error:
                    return None
                else:
                    raise AttributeError(f"Attribute {attribute_name} doesn't exist in input data and must therefore be defined")
            elif not skip_warning:
                logging.warning(f"DeprecationWarning: Attribute names without '_default' suffix are deprecated. \nChange for {attribute_name} of attributes in path {self.folder_path}")
        else:
            attribute_name = attribute_name + "_default"
        return attribute_name

    def extract_yearly_variation(self, file_name, index_sets, unit_category):
        """ reads the yearly variation of a time dependent quantity

        :param file_name: name of selected file.
        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param scenario: scenario name
        """
        # remove intra-yearly time steps from index set and add inter-yearly time steps
        index_sets = copy.deepcopy(index_sets)
        index_sets.remove("set_time_steps")
        index_sets.append("set_time_steps_yearly")
        # add Yearly_variation to file_name
        file_name += "_yearly_variation"
        # read input data
        f_name, scenario_factor = self.scenario_dict.get_param_file(self.element.name, file_name)
        df_input = self.read_input_data(f_name)
        if f_name != file_name and df_input is None:
            logging.info(f"{f_name} is missing from {self.folder_path}. {file_name} is used as input file")
            df_input = self.read_input_data(file_name)
        if df_input is not None:
            df_output, default_value, index_name_list = self.create_default_output(index_sets, unit_category, file_name=file_name, manual_default_value=1)
            # set yearly variation attribute to df_output
            name_yearly_variation = file_name
            df_output = self.extract_general_input_data(df_input, df_output, file_name, index_name_list, default_value, time_steps="set_time_steps_yearly")
            # apply the scenario_factor
            df_output = df_output * scenario_factor
            setattr(self, name_yearly_variation, df_output)

    def extract_locations(self, extract_nodes=True, extract_coordinates=False):
        """ reads input data to extract nodes or edges.

        :param extract_nodes: boolean to switch between nodes and edges
        :param extract_coordinates: boolean to switch between nodes and nodes + coordinates
        """
        if extract_nodes:
            set_nodes_config = self.system["set_nodes"]
            df_nodes_w_coords = self.read_input_data("set_nodes")
            if extract_coordinates:
                if len(set_nodes_config) != 0:
                    df_nodes_w_coords = df_nodes_w_coords[df_nodes_w_coords["node"].isin(set_nodes_config)]
                return df_nodes_w_coords
            else:
                set_nodes_input = df_nodes_w_coords["node"].to_list()
                # if no nodes specified in system, use all nodes
                if len(set_nodes_config) == 0 and not len(set_nodes_input) == 0:
                    self.system["set_nodes"] = set_nodes_input
                    set_nodes_config = set_nodes_input
                else:
                    assert len(set_nodes_config) > 1, f"ZENx is a spatially distributed model. Please specify at least 2 nodes."
                    _missing_nodes = list(set(set_nodes_config).difference(set_nodes_input))
                    assert len(_missing_nodes) == 0, f"The nodes {_missing_nodes} were declared in the config but do not exist in the input file {self.folder_path + 'set_nodes'}"
                if not isinstance(set_nodes_config, list):
                    set_nodes_config = set_nodes_config.to_list()
                set_nodes_config.sort()
                return set_nodes_config
        else:
            set_edges_input = self.read_input_data("set_edges")
            self.energy_system.optimization_setup.input_data_checks.check_single_directed_edges(set_edges_input=set_edges_input)
            if set_edges_input is not None:
                set_edges = set_edges_input[(set_edges_input["node_from"].isin(self.energy_system.set_nodes)) & (set_edges_input["node_to"].isin(self.energy_system.set_nodes))]
                set_edges = set_edges.set_index("edge")
                return set_edges
            else:
                return None

    def extract_carriers(self, carrier_type, unit_category):
        """ reads input data and extracts conversion carriers

        :return carrier_list: list with input, output or reference carriers of technology """
        assert carrier_type in ["input_carrier", "output_carrier", "reference_carrier"], "carrier type must be either input_carrier, output_carrier, or reference_carrier"
        carrier_string = self.extract_attribute(carrier_type, unit_category, skip_warning=True)
        if type(carrier_string) == str:
            carrier_list = carrier_string.strip().split(" ")
        else:
            carrier_list = []
        assert carrier_type != "reference_carrier" or len(carrier_list) == 1, f"reference_carrier must be a single carrier, but {carrier_list} are given for {self.element.name}"
        return carrier_list

    def extract_set_technologies_existing(self, storage_energy=False):
        """ reads input data and creates setExistingCapacity for each technology

        :param storage_energy: boolean if existing energy capacity of storage technology (instead of power)
        :return set_technologies_existing: return set existing technologies"""
        #TODO merge changes in extract input data and optimization setup
        set_technologies_existing = np.array([0])
        if self.analysis["use_capacities_existing"]:
            if storage_energy:
                _energy_string = "_energy"
            else:
                _energy_string = ""

            # here we ignore the factor
            f_name, _ = self.scenario_dict.get_param_file(self.element.name, f"capacity_existing{_energy_string}")
            df_input = self.read_input_data(f_name)
            if df_input is None:
                return [0]
            if self.element.name in self.system["set_transport_technologies"]:
                location = "edge"
            else:
                location = "node"
            _max_node_count = df_input[location].value_counts().max()
            if _max_node_count is not np.nan:
                set_technologies_existing = np.arange(0, _max_node_count)

        return set_technologies_existing

    def extract_lifetime_existing(self, file_name, index_sets):
        """ reads input data and restructures the dataframe to return (multi)indexed dict

        :param file_name:  name of selected file
        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param scenario: scenario name
        :return df_output: return existing capacity and existing lifetime """
        index_list, index_name_list = self.construct_index_list(index_sets, None)
        multiidx = pd.MultiIndex.from_product(index_list, names=index_name_list)
        df_output = pd.Series(index=multiidx, data=0)
        # if no existing capacities
        if not self.analysis["use_capacities_existing"]:
            return df_output

        f_name, scenario_factor = self.scenario_dict.get_param_file(self.element.name, file_name)
        if f"{f_name}.csv" in os.listdir(self.folder_path):
            df_input = self.read_input_data(f_name)
            # fill output dataframe
            df_output = self.extract_general_input_data(df_input, df_output, "year_construction", index_name_list, default_value=0, time_steps=None)
            # get reference year
            reference_year = self.system["reference_year"]
            # calculate remaining lifetime
            df_output[df_output > 0] = - reference_year + df_output[df_output > 0] + self.element.lifetime[0]
        # apply scenario factor
        return df_output*scenario_factor

    def extract_pwa_data(self, variable_type):
        """ reads input data and restructures the dataframe to return (multi)indexed dict

        :param variable_type: technology approximation type
        :return pwa_dict: dictionary with pwa parameters """
        # attribute names
        if variable_type == "capex":
            attribute_name = "capex_specific"
            index_sets = ["set_nodes", "set_time_steps_yearly"]
            time_steps = "set_time_steps_yearly"
            unit_category = {"money": 1, "energy_quantity": -1, "time": -1}
        elif variable_type == "conversion_factor":
            attribute_name = "conversion_factor"
            index_sets = ["set_nodes", "set_time_steps"]
            time_steps = "set_base_time_steps_yearly"
            unit_category = {"energy_quantity": 0}
        else:
            raise KeyError(f"variable type {variable_type} unknown.")
        # import all input data
        df_input_nonlinear = self.read_pwa_files(variable_type, file_type="nonlinear_")
        df_input_breakpoints = self.read_pwa_files(variable_type, file_type="breakpoints_pwa_")
        df_input_linear = self.read_pwa_files(variable_type)
        df_linear_exist = self.exists_attribute(attribute_name, unit_category)
        assert (df_input_nonlinear is not None and df_input_breakpoints is not None) or df_linear_exist or df_input_linear is not None, f"Neither pwa nor linear data exist for {variable_type} of {self.element.name}"
        # if nonlinear
        if (df_input_nonlinear is not None and df_input_breakpoints is not None):
            # select data
            pwa_dict = {}
            # extract all data values
            nonlinear_values = {}

            if variable_type == "capex":
                # make absolute capex
                df_input_nonlinear["capex"] = df_input_nonlinear["capex"] * df_input_nonlinear["capacity"]
            for column in df_input_nonlinear.columns:
                nonlinear_values[column] = df_input_nonlinear[column].to_list()

            # assert that breakpoint variable (x variable in nonlinear input)
            assert df_input_breakpoints.columns[0] in df_input_nonlinear.columns, \
                f"breakpoint variable for pwa '{df_input_breakpoints.columns[0]}' is not in nonlinear variables [{df_input_nonlinear.columns}]"
            breakpoint_variable = df_input_breakpoints.columns[0]
            breakpoints = df_input_breakpoints[breakpoint_variable].to_list()

            pwa_dict[breakpoint_variable] = breakpoints
            pwa_dict["pwa_variables"] = []  # select only those variables that are modeled as pwa
            pwa_dict["bounds"] = {}  # save bounds of variables
            linear_dict = {}
            # min and max total capacity of technology
            min_capacity_tech, max_capacity_tech = (0, min(max(self.element.capacity_limit.values), max(breakpoints)))
            for value_variable in nonlinear_values:
                if value_variable == breakpoint_variable:
                    pwa_dict["bounds"][value_variable] = (min_capacity_tech, max_capacity_tech)
                else:
                    # conduct linear regress
                    linear_regress_object = linregress(nonlinear_values[breakpoint_variable],
                                                       nonlinear_values[value_variable])
                    # calculate relative intercept (intercept/slope) if slope != 0
                    if linear_regress_object.slope != 0:
                        relative_intercept = np.abs(linear_regress_object.intercept / linear_regress_object.slope)
                    else:
                        relative_intercept = np.abs(linear_regress_object.intercept)
                    # check if to a reasonable degree linear
                    if relative_intercept <= self.solver["linear_regression_check"]["eps_intercept"] \
                            and linear_regress_object.rvalue >= self.solver["linear_regression_check"]["epsRvalue"]:
                        # model as linear function
                        slope_lin_reg = linear_regress_object.slope
                        linear_dict[value_variable] = \
                            self.create_default_output(index_sets=index_sets, unit_category=unit_category, time_steps=time_steps,
                                                   manual_default_value=slope_lin_reg)[0]
                    else:
                        # model as pwa function
                        pwa_dict[value_variable] = list(np.interp(breakpoints, nonlinear_values[breakpoint_variable],
                                                                  nonlinear_values[value_variable]))
                        pwa_dict["pwa_variables"].append(value_variable)
                        # save bounds
                        values_between_bounds = [pwa_dict
                             [value_variable][idxBreakpoint] for idxBreakpoint, breakpoint in enumerate(breakpoints)
                                if breakpoint >= min_capacity_tech and breakpoint <= max_capacity_tech
                        ]
                        values_between_bounds.extend(list(
                            np.interp([min_capacity_tech, max_capacity_tech], breakpoints, pwa_dict[value_variable])))
                        pwa_dict["bounds"][value_variable] = (min(values_between_bounds), max(values_between_bounds))
            # pwa
            if (len(pwa_dict["pwa_variables"]) > 0 and len(linear_dict) == 0):
                is_pwa = True
                return pwa_dict, is_pwa
            # linear
            elif len(linear_dict) > 0 and len(pwa_dict["pwa_variables"]) == 0:
                is_pwa = False
                linear_dict = pd.DataFrame.from_dict(linear_dict)
                linear_dict.columns.name = "carrier"
                linear_dict = linear_dict.stack()
                conversion_factor_levels = [linear_dict.index.names[-1]] + linear_dict.index.names[:-1]
                linear_dict = linear_dict.reorder_levels(conversion_factor_levels)
                return linear_dict, is_pwa
            # no dependent carrier
            elif len(nonlinear_values) == 1:
                is_pwa = False
                return None, is_pwa
            else:
                raise NotImplementedError(
                    f"There are both linearly and nonlinearly modeled variables in {variable_type} of {self.element.name}. Not yet implemented")
        # linear
        else:
            is_pwa = False
            linear_dict = {}
            if variable_type == "capex":
                linear_dict["capex"] = self.extract_input_data(attribute_name, index_sets=index_sets, unit_category=unit_category, time_steps=time_steps)
                return linear_dict, is_pwa
            else:
                dependent_carrier = list(set(self.element.input_carrier + self.element.output_carrier).difference(
                    self.element.reference_carrier))
                if not dependent_carrier:
                    return None, is_pwa
                if df_input_linear is None:
                    if len(dependent_carrier) == 1:
                        linear_dict[dependent_carrier[0]] = self.extract_input_data(attribute_name, index_sets=index_sets, unit_category=unit_category, time_steps=time_steps)
                    else:
                        raise AssertionError(f"input file for linear_conversion_factor could not be imported.")
                else:
                    df_output, default_value, index_name_list = self.create_default_output(index_sets, unit_category, attribute_name, time_steps=time_steps)
                    common_index = list(set(index_name_list).intersection(df_input_linear.columns))
                    # if only one dependent carrier and no carrier in columns
                    if len(dependent_carrier) == 1 and len(df_input_linear.columns.intersection(dependent_carrier)) == 0:
                        linear_dict[dependent_carrier[0]] = self.extract_general_input_data(df_input_linear, df_output,
                                                                               "conversion_factor",
                                                                               index_name_list, default_value,
                                                                               time_steps=time_steps).copy(deep=True)
                    else:
                        for carrier in dependent_carrier:
                            if common_index:
                                df_input_carrier = df_input_linear[common_index + [carrier]]
                                linear_dict[carrier] = self.extract_general_input_data(df_input_carrier, df_output, "conversion_factor", index_name_list, default_value, time_steps=time_steps).copy(deep=True)
                            else:
                                linear_dict[carrier] = df_output.copy()
                                linear_dict[carrier].loc[:] = df_input_linear[carrier].squeeze()
                linear_dict = pd.DataFrame.from_dict(linear_dict)
                linear_dict.columns.name = "carrier"
                linear_dict = linear_dict.stack()
                conversion_factor_levels = [linear_dict.index.names[-1]] + linear_dict.index.names[:-1]
                linear_dict = linear_dict.reorder_levels(conversion_factor_levels)
                # extract yearly variation
                self.extract_yearly_variation(attribute_name, index_sets, unit_category)
                return linear_dict, is_pwa

    def read_pwa_files(self, variable_type, file_type=str()):
        """ reads pwa files

        :param variable_type: technology approximation type
        :param file_type: either breakpointsPWA, linear, or nonlinear
        :return df_input: raw input file"""
        df_input = self.read_input_data(file_type + variable_type)
        if df_input is not None:
            if "unit" in df_input.values:
                columns = df_input.iloc[-1][df_input.iloc[-1] != "unit"].dropna().index
            else:
                columns = df_input.columns
            df_input_units = df_input[columns].iloc[-1]
            #save conversion factor and nonlinear capex units for unit consistency checks
            if not df_input_units.empty and file_type != "breakpoints_pwa_":
                if file_type == "nonlinear_":
                    if variable_type == "conversion_factor":
                        self.element.units_conversion_factor_files = {"nonlinear": df_input_units}
                    else:
                        self.element.units_nonlinear_capex_files = df_input_units
                elif variable_type == "conversion_factor":
                    self.element.units_conversion_factor_files = {"linear": df_input_units}
            df_input = df_input.iloc[:-1]
            df_input_multiplier = df_input_units.apply(lambda unit: self.unit_handling.get_unit_multiplier(unit, attribute_name=variable_type))
            df_input = df_input.apply(lambda column: pd.to_numeric(column, errors='ignore'))
            df_input[columns] = df_input[columns] * df_input_multiplier
        return df_input

    def create_default_output(self, index_sets, unit_category, file_name=None, time_steps=None, manual_default_value=None):
        """ creates default output dataframe

        :param file_name: name of selected file.
        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param time_steps: specific time_steps of element
        :param manual_default_value: if given, use manual_default_value instead of searching for default value in attributes.csv"""
        # select index
        index_list, index_name_list = self.construct_index_list(index_sets, time_steps)
        # create pd.MultiIndex and select data
        if index_sets:
            index_multi_index = pd.MultiIndex.from_product(index_list, names=index_name_list)
        else:
            index_multi_index = pd.Index([0])
        # use distances computed with node coordinates as default values
        if file_name == "distance":
            default_name = file_name
            default_value = self.extract_attribute(default_name, unit_category)
            default_value["value"] = manual_default_value
        elif manual_default_value:
            default_value = {"value": manual_default_value, "multiplier": 1}
            default_name = None
        else:
            default_name = file_name
            default_value = self.extract_attribute(default_name, unit_category)

        # create output Series filled with default value
        if default_value is None:
            df_output = pd.Series(index=index_multi_index, dtype=float)
        # use distances computed with node coordinates as default values
        elif file_name == "distance":
            df_output = pd.Series(index=index_multi_index, dtype=float)
            for key, value in default_value["value"].items():
                df_output[key] = value
        else:
            df_output = pd.Series(index=index_multi_index, data=default_value["value"], dtype=float)
        # save unit of attribute of element converted to base unit
        self.save_unit_of_attribute(default_name)
        return df_output, default_value, index_name_list

    def save_unit_of_attribute(self, file_name):
        """ saves the unit of an attribute, converted to the base unit """
        # if numerics analyzed
        if self.solver["analyze_numerics"]:
            attributes, _ = self.scenario_dict.get_default(self.element.name, file_name)
            df_input = self.read_input_data(attributes).set_index("index").squeeze(axis=1)
            # get attribute
            if file_name:
                attribute_name = self.adapt_attribute_name(file_name, df_input)
                input_unit = df_input.loc[attribute_name, "unit"]
            else:
                input_unit = np.nan
            self.unit_handling.set_base_unit_combination(input_unit=input_unit, attribute=(self.element.name, file_name))

    def save_values_of_attribute(self, df_output, file_name):
        """ saves the values of an attribute

        :param df_output: default output dataframe
        :param file_name: name of selected file.
        """
        # if numerics analyzed
        if self.solver["analyze_numerics"]:
            if file_name:
                df_output_reduced = df_output[(df_output != 0) & (df_output.abs() != np.inf)]
                if not df_output_reduced.empty:
                    self.unit_handling.set_attribute_values(df_output=df_output_reduced, attribute=(self.element.name, file_name))

    def construct_index_list(self, index_sets, time_steps):
        """ constructs index list from index sets and returns list of indices and list of index names

        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param time_steps: specific time_steps of element
        :return index_list: list of indices
        :return index_name_list: list of name of indices
        """
        index_list = []
        index_name_list = []
        # add rest of indices
        for index in index_sets:
            index_name_list.append(self.index_names[index])
            if "set_time_steps" in index and time_steps:
                index_list.append(getattr(self.energy_system, time_steps))
            elif index == "set_technologies_existing":
                index_list.append(self.element.set_technologies_existing)
            elif index in self.system:
                index_list.append(self.system[index])
            elif hasattr(self.energy_system, index):
                index_list.append(getattr(self.energy_system, index))
            else:
                raise AttributeError(f"Index '{index}' cannot be found.")
        return index_list, index_name_list

    def exists_attribute(self, file_name, unit_category, column=None):
        """ checks if default value or timeseries of an attribute exists in the input data

        :param file_name: name of selected file
        :param column: select specific column
        """
        # check if default value exists
        if column:
            default_name = column
        else:
            default_name = file_name
        default_value = self.extract_attribute(default_name, unit_category, check_if_exists=True)

        if default_value is None or math.isnan(default_value["value"]):  # if no default value exists or default value is nan
            _dfInput = self.read_input_data(file_name)
            return (_dfInput is not None)
        elif default_value and not math.isnan(default_value["value"]):  # if default value exists and is not nan
            return True
        else:
            return False

    def convert_real_to_generic_time_indices(self, df_input, time_steps, file_name, index_name_list):
        """convert yearly time indices to generic time indices

        :param df_input: raw input dataframe
        :param time_steps: specific time_steps of element
        :param file_name: name of selected file
        :param index_name_list: list of name of indices
        :return df_input: input dataframe with generic time indices
        """
        # check if input data is time-dependent and has yearly time steps
        idx_name_year = self.index_names["set_time_steps_yearly"]
        if time_steps == "set_time_steps_yearly" or time_steps == "set_time_steps_yearly_entire_horizon":
            # check if temporal header of input data is still given as 'time' instead of 'year'
            if "time" in df_input.axes[1]:
                logging.warning(
                    f"DeprecationWarning: The column header 'time' (used in {file_name}) will not be supported for input data with yearly time steps any longer! Use the header 'year' instead")
                df_input = df_input.rename(
                    {self.index_names["set_time_steps"]: self.index_names["set_time_steps_yearly"]}, axis=1)
            # does not contain annual index
            elif idx_name_year not in df_input.axes[1]:
                idx_name_list = [idx for idx in index_name_list if idx != idx_name_year]
                # no other index called, return original time series
                if not idx_name_list:
                    return df_input
                df_input = df_input.set_index(idx_name_list)
                df_input = df_input.rename(columns={col: int(col) for col in df_input.columns if col.isnumeric()})
                requested_index_values = set(getattr(self.energy_system, time_steps))
                requested_index_values_years = set(self.energy_system.set_time_steps_years)
                requested_index_values_in_columns = requested_index_values.intersection(df_input.columns)
                requested_index_values_years_in_columns = requested_index_values_years.intersection(df_input.columns)
                if not requested_index_values_in_columns and not requested_index_values_years_in_columns:
                    return df_input.reset_index()
                elif requested_index_values_in_columns:
                    requested_index_values = requested_index_values_in_columns
                else:
                    requested_index_values = requested_index_values_years_in_columns
                df_input.columns = df_input.columns.set_names(idx_name_year)
                df_input = df_input[list(requested_index_values)].stack()
                df_input = df_input.reset_index()
            # check if input data is still given with generic time indices
            temporal_header = self.index_names["set_time_steps_yearly"]
            if max(df_input.loc[:, temporal_header]) < self.analysis["earliest_year_of_data"]:
                logging.warning(
                    f"DeprecationWarning: Generic time indices (used in {file_name}) will not be supported for input data with yearly time steps any longer! Use the corresponding years (e.g. 2022,2023,...) as time indices instead")
                return df_input
            # assert that correct temporal index_set to get corresponding index_name is given (i.e. set_time_steps_yearly for input data with yearly time steps)(otherwise extract_general_input_data() will find a missing_index)
            assert temporal_header in index_name_list, f"Input data with yearly time steps and therefore the temporal header 'year' needs to be extracted with index_sets=['set_time_steps_yearly'] instead of index_sets=['set_time_steps']"
            # set index
            index_names_column = df_input.columns.intersection(index_name_list).to_list()
            df_input = df_input.set_index(index_names_column)
            if df_input.index.nlevels == 1:
                combined_index = df_input.index.union(self.energy_system.set_time_steps_years)
                is_single_index = True
            else:
                index_list = []
                for index_name in index_names_column:
                    if index_name == temporal_header:
                        index_list.append(df_input.index.get_level_values(index_name).unique().union(
                            self.energy_system.set_time_steps_years))
                    else:
                        index_list.append(df_input.index.get_level_values(index_name).unique())
                combined_index = pd.MultiIndex.from_product(index_list, names=index_names_column).sort_values()
                is_single_index = False
            df_input_temp = pd.DataFrame(index=combined_index, columns=df_input.columns)
            common_index = df_input.index.intersection(combined_index)
            df_input_temp.loc[common_index] = df_input.loc[common_index]
            # df_input_temp.loc[df_input.index] = df_input
            df_input = df_input_temp.astype(float)
            # interpolate missing data
            file_names_int_off = []
            if self.energy_system.parameters_interpolation_off is not None:
                file_names_int_off = self.energy_system.parameters_interpolation_off.values
            if file_name not in file_names_int_off:
                parameters = df_input.axes[1]
                for param in parameters:
                    if param not in index_names_column and df_input[param].isna().any():
                        if is_single_index:
                            df_input[param] = df_input[param].astype(float).interpolate(method="index")
                        else:
                            df_input_temp = df_input[param].unstack(df_input.index.names.difference([temporal_header]))
                            df_input[param] = df_input_temp.interpolate(method="index", axis=0).stack().reorder_levels(
                                df_input.index.names)
            else:
                logging.info(f"Parameter {file_name} data won't be interpolated to cover years without given values")
            df_input = df_input.reset_index()
            # remove data of years that won't be simulated
            df_input = df_input[df_input[temporal_header].isin(self.energy_system.set_time_steps_years)]
            # convert yearly time indices to generic ones
            year2step = {year: step for year, step in zip(self.energy_system.set_time_steps_years, getattr(self.energy_system, time_steps))}
            df_input[temporal_header] = df_input[temporal_header].apply(lambda year: year2step[year])
        return df_input

    @staticmethod
    def extract_from_input_without_missing_index(df_input, index_name_list, file_name):
        """ extracts the demanded values from Input dataframe and reformulates dataframe

        :param df_input: raw input dataframe
        :param index_name_list: list of name of indices
        :param file_name: name of selected file
        :return df_input: reformulated input dataframe
        """
        if index_name_list:
            df_input = df_input.set_index(index_name_list)
        assert len(df_input.columns) == 1, f"Input file for {file_name} has more than one value column: {df_input.columns.to_list()}"
        df_input = df_input.squeeze(axis=1)
        return df_input

    @staticmethod
    def extract_from_input_with_missing_index(df_input, df_output, index_name_list, file_name, missing_index):
        """ extracts the demanded values from Input dataframe and reformulates dataframe if the index is missing.
        Either, the missing index is the column of df_input, or it is actually missing in df_input.
        Then, the values in df_input are extended to all missing index values.

        :param df_input: raw input dataframe
        :param df_output: default output dataframe
        :param index_name_list: list of name of indices
        :param file_name: name of selected file
        :param missing_index: missing index in df_input
        :return df_input: reformulated input dataframe
        """
        index_name_list.remove(missing_index)
        if not index_name_list:
            # assert that single value
            assert df_input.size == 1, f"Cannot establish unique values for file {file_name} because of too many columns or not overlapping index"
            val_input = df_input.squeeze()
            df_output[:] = val_input
            df_input = df_output.copy()
            return df_input
        df_input = df_input.set_index(index_name_list)
        # missing index values
        requested_index_values = set(df_output.index.get_level_values(missing_index))
        # the missing index is the columns of df_input
        requested_index_values_in_columns = requested_index_values.intersection(df_input.columns)
        if requested_index_values_in_columns:
            requested_index_values = requested_index_values_in_columns
            df_input.columns = df_input.columns.set_names(missing_index)
            df_input = df_input[list(requested_index_values)].stack()
            df_input = df_input.reorder_levels(df_output.index.names)
        # the missing index does not appear in df_input
        # the values in df_input are extended to all missing index values
        else:
            df_input_index_temp = pd.MultiIndex.from_product([df_input.index, requested_index_values], names=df_input.index.names + [missing_index])
            df_input_temp = pd.Series(index=df_input_index_temp, dtype=float)
            if isinstance(df_input, pd.Series):
                df_input = df_input.to_frame()
            if df_input.shape[1] == 1:
                df_input = df_input.loc[df_input_index_temp.get_level_values(df_input.index.names[0])].squeeze(axis=1)
            else:
                assert df_input_temp.index.names[-1] != "time", f"Only works if columns contain time index and not for {df_input_temp.index.names[-1]}"
                df_input = df_input_temp.to_frame().apply(lambda row: df_input.loc[row.name[0:-1], str(row.name[-1])], axis=1)
            df_input.index = df_input_temp.index
            df_input = df_input.reorder_levels(order=df_output.index.names)
            if isinstance(df_input, pd.DataFrame):
                df_input = df_input.squeeze(axis=1)
        return df_input

    @staticmethod
    def extract_from_input_for_capacities_existing(df_input, df_output, index_name_list, column, missing_index):
        """ extracts the demanded values from input dataframe if extracting existing capacities

        :param df_input: raw input dataframe
        :param df_output: default output dataframe
        :param index_name_list: list of name of indices
        :param column: select specific column
        :param missing_index: missing index in df_input
        :return df_output: filled output dataframe
        """
        index_name_list.remove(missing_index)
        df_input = df_input.set_index(index_name_list)
        set_location = df_input.index.unique()
        for location in set_location:
            if location in df_output.index.get_level_values(index_name_list[0]):
                values = df_input[column].loc[location].tolist()
                if isinstance(values, int) or isinstance(values, float):
                    index = [0]
                else:
                    index = list(range(len(values)))
                df_output.loc[location, index] = values
        return df_output
