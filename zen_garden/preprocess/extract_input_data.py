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
import json
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
        self.index_names = self.analysis['header_data_inputs']
        # load attributes file
        self.attribute_dict = self.load_attribute_file()

    def extract_input_data(self, file_name, index_sets, unit_category, time_steps=None, subelement=None):
        """ reads input data and restructures the dataframe to return (multi)indexed dict

        :param file_name: name of selected file.
        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param unit_category: dict defining the dimensions of the parameter's unit
        :param time_steps: string specifying time_steps
        :param subelement: string specifying dependent element
        :return: dictionary with attribute values """

        # generic time steps
        yearly_variation = False
        if not time_steps:
            time_steps = "set_base_time_steps"
        # if time steps are the yearly base time steps
        elif time_steps == "set_base_time_steps_yearly":
            yearly_variation = True
            self.extract_yearly_variation(file_name, index_sets)

        # if existing capacities and existing capacities not used
        if (file_name == "capacity_existing" or file_name == "capacity_existing_energy") and not self.system["use_capacities_existing"]:
            df_output, *_ = self.create_default_output(index_sets, unit_category, file_name=file_name, time_steps=time_steps, manual_default_value=0)
            return df_output
        # use distances computed with node coordinates as default values
        elif file_name == "distance":
            df_output, default_value, index_name_list = self.create_default_output(index_sets, unit_category, file_name=file_name, time_steps=time_steps, manual_default_value=self.energy_system.set_haversine_distances_edges)
        else:
            df_output, default_value, index_name_list = self.create_default_output(index_sets, unit_category, file_name=file_name, time_steps=time_steps,subelement=subelement)
        # read input file
        f_name, scenario_factor = self.scenario_dict.get_param_file(self.element.name, file_name)
        df_input = self.read_input_csv(f_name)
        if f_name != file_name and yearly_variation and df_input is None:
            logging.info(f"{f_name} for current scenario is missing from {self.folder_path}. {file_name} is used as input file")
            df_input = self.read_input_csv(file_name)

        assert (df_input is not None or default_value is not None), f"input file for attribute {file_name} could not be imported and no default value is given."
        if df_input is not None and not df_input.empty:
            # get subelement dataframe
            if subelement is not None and subelement in df_input.columns:
                cols = df_input.columns.intersection(index_name_list + [subelement])
                df_input = df_input[cols]
            # fill output dataframe
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

        # check for duplicate indices
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

    def read_input_csv(self, input_file_name):
        """ reads input data and returns raw input dataframe

        :param input_file_name: name of selected file
        :return df_input: pd.DataFrame with input data """

        # append .csv suffix
        input_file_name += ".csv"

        # select data
        file_names = os.listdir(self.folder_path)
        if input_file_name in file_names:
            df_input = pd.read_csv(os.path.join(self.folder_path, input_file_name), header=0, index_col=None)
            # check for header name duplicates (pd.read_csv() adds a dot and a number to duplicate headers)
            if any("." in col for col in df_input.columns):
                raise AssertionError(f"The input data file {input_file_name} at {self.folder_path} contains two identical header names.")
            return df_input
        else:
            return None

    def load_attribute_file(self, filename="attributes"):
        """
        loads attribute file. Either as csv (old version) or json (new version)
        :param filename: name of attributes file, default is 'attributes'
        :return: attribute_dict
        """
        if os.path.exists(self.folder_path / f"{filename}.json"):
            attribute_dict = self._load_attribute_file_json(filename=filename)
        # extract csv
        elif os.path.exists(self.folder_path / f"{filename}.csv"):
            raise NotImplementedError(f"The .csv format for attributes is deprecated ({filename} of {self.element.name}). Use .json instead.")
        else:
            raise FileNotFoundError(f"Attributes file does not exist for {self.element.name}")
        return attribute_dict

    def _load_attribute_file_json(self, filename):
        """
        loads json attributes file
        :param filename:
        :return: attributes
        """
        file_path = self.folder_path / f"{filename}.json"
        with open(file_path, "r") as file:
            data = json.load(file)
        attribute_dict = {}
        if type(data) == list:
            logging.warning("DeprecationWarning: The list format in attributes.json [{...}] is deprecated. Use a dict format instead {...}.")
            for item in data:
                for k, v in item.items():
                    if type(v) == list:
                        attribute_dict[k] = {sk: sv for d in v for sk, sv in d.items()}
                    else:
                        attribute_dict[k] = v
        else:
            for k, v in data.items():
                if type(v) == list:
                    attribute_dict[k] = {sk: sv for d in v for sk, sv in d.items()}
                else:
                    attribute_dict[k] = v
        return attribute_dict

    def get_attribute_dict(self, attribute_name):
        """ get attribute dict and factor for attribute

        :param attribute_name: name of selected attribute
        :return attribute_dict: attribute dict
        :return factor: factor for attribute """
        if self.scenario_dict is not None:
            filename, factor = self.scenario_dict.get_default(self.element.name, attribute_name)
        else:
            filename = "attributes"
            factor = 1
        if filename != "attributes":
            attribute_dict = self.load_attribute_file(filename)
        else:
            attribute_dict = self.attribute_dict
        return attribute_dict, factor

    def extract_attribute(self, attribute_name, unit_category, return_unit=False,subelement=None):
        """ reads input data and restructures the dataframe to return (multi)indexed dict

        :param attribute_name: name of selected attribute
        :param unit_category: dict defining the dimensions of the parameter's unit
        :param return_unit: only returns unit
        :param subelement: dependent element for which data is extracted
        :return: attribute value and multiplier
        :return: unit of attribute """
        attribute_dict, factor = self.get_attribute_dict(attribute_name)
        attribute_value, attribute_unit = self._extract_attribute_value(attribute_name,attribute_dict)
        if subelement is not None:
            assert subelement in attribute_value.keys(), f"{subelement} not in {attribute_name} of {self.element.name}"
            attribute_unit = attribute_value[subelement]["unit"]
            attribute_value = attribute_value[subelement]["default_value"]
        if return_unit:
            return attribute_unit
        if attribute_unit is None:
            return attribute_value
        if attribute_value is not None:
            multiplier, attribute_unit_in_base_units = self.unit_handling.convert_unit_into_base_units(attribute_unit, get_multiplier=True, attribute_name=attribute_name, path=self.folder_path)
            # don't convert unit of conversion factor to base units since e.g. kWh/kWh would become 1 (however, conversion factors' unit consistency must be checked with the corresponding carriers)
            if attribute_name == "conversion_factor":
                if attribute_name not in self.element.units:
                    self.element.units[attribute_name] = {}
                self.element.units[attribute_name][subelement] = {"unit_category": unit_category, "unit": attribute_unit}
            elif attribute_name == "retrofit_flow_coupling_factor":
                self.element.units[attribute_name] = {str(self.element.reference_carrier[0]): {"unit_category": unit_category, "unit": attribute_unit}}
            # don't try to save input-/output carrier if they don't exist for a conversion technology
            elif not (pd.isna(attribute_value) and attribute_name in ["input_carrier", "output_carrier"]):
                self.element.units[attribute_name] = {"unit_category": unit_category, "unit_in_base_units": attribute_unit_in_base_units}
            try:
                attribute = {"value": float(attribute_value) * multiplier * factor, "multiplier": multiplier}
                return attribute
            except ValueError:
                if factor != 1:
                    logging.warning(f"WARNING: Attribute {attribute_name} of {self.element.name} is not a number "
                                    f"but has custom factor {factor}, factor will be ignored...")
                attribute = attribute_value
                return attribute
        else:
            return None

    def _extract_attribute_value(self,attribute_name,attribute_dict):
        """
        reads attribute value from dict
        :param attribute_name: name of selected attribute
        :param attribute_dict: name of selected attribute
        :return: attribute value, attribute unit
        """
        if attribute_name not in attribute_dict:
            raise AttributeError(f"Attribute {attribute_name} does not exist in input data of {self.element.name}")
        try:
            attribute_value = float(attribute_dict[attribute_name]["default_value"])
            attribute_unit = attribute_dict[attribute_name]["unit"]
        # for string attributes
        except ValueError:
            attribute_value = attribute_dict[attribute_name]["default_value"]
            attribute_unit = attribute_dict[attribute_name]["unit"]
        # for list of attributes
        except (TypeError, KeyError):
            if "default_value" in attribute_dict[attribute_name]:
                attribute_value = attribute_dict[attribute_name]["default_value"]
            else:
                attribute_value = attribute_dict[attribute_name]
            attribute_unit = None
        return attribute_value,attribute_unit

    def extract_yearly_variation(self, file_name, index_sets):
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
        df_input = self.read_input_csv(f_name)
        if f_name != file_name and df_input is None:
            logging.info(f"{f_name} is missing from {self.folder_path}. {file_name} is used as input file")
            df_input = self.read_input_csv(file_name)
        if df_input is not None:
            df_output, default_value, index_name_list = self.create_default_output(index_sets, unit_category=None, file_name=file_name, manual_default_value=1)
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
            df_nodes_w_coords = self.read_input_csv("set_nodes")
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
            set_edges_input = self.read_input_csv("set_edges")
            self.energy_system.optimization_setup.input_data_checks.check_single_directed_edges(set_edges_input=set_edges_input)
            if set_edges_input is not None:
                set_edges = set_edges_input[(set_edges_input["node_from"].isin(self.energy_system.set_nodes)) & (set_edges_input["node_to"].isin(self.energy_system.set_nodes))]
                set_edges = set_edges.set_index("edge")
                return set_edges
            else:
                raise FileNotFoundError(f"Input file set_edges.csv is missing from {self.folder_path}")

    def extract_carriers(self, carrier_type):
        """ reads input data and extracts conversion carriers

        :return carrier_list: list with input, output or reference carriers of technology """
        assert carrier_type in ["input_carrier", "output_carrier", "reference_carrier", "retrofit_reference_carrier"], "carrier type must be either input_carrier, output_carrier,retrofit_reference_carrier, or reference_carrier"
        carrier_list = self.extract_attribute(carrier_type, unit_category=None)
        assert carrier_type != "reference_carrier" or len(carrier_list) == 1, f"Reference_carrier must be a single carrier, but {carrier_list} are given for {self.element.name}"
        if carrier_list == [""]:
            carrier_list = []
        return carrier_list

    def extract_retrofit_base_technology(self):
        """ extract base technologies for retrofitting technology

        :return base_technology: return base technology of retrofit technology """
        attribute_name = "retrofit_flow_coupling_factor"
        technology_type = "base_technology"
        attribute_dict, _ = self.get_attribute_dict(attribute_name)
        base_technology = attribute_dict[attribute_name][technology_type]
        if type(base_technology) == str:
            base_technology = base_technology.strip().split(" ")
        assert len(base_technology) == 1, f"retrofit base technology must be a single technology, but {base_technology} are given for {self.element.name}"
        return base_technology

    def extract_set_technologies_existing(self, storage_energy=False):
        """ reads input data and creates setExistingCapacity for each technology

        :param storage_energy: boolean if existing energy capacity of storage technology (instead of power)
        :return set_technologies_existing: return set existing technologies"""
        #TODO merge changes in extract input data and optimization setup
        set_technologies_existing = np.array([0])
        if self.system["use_capacities_existing"]:
            if storage_energy:
                _energy_string = "_energy"
            else:
                _energy_string = ""

            # here we ignore the factor
            f_name, _ = self.scenario_dict.get_param_file(self.element.name, f"capacity_existing{_energy_string}")
            df_input = self.read_input_csv(f_name)
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
        df_output = pd.Series(index=multiidx, data=0,dtype=int)
        # if no existing capacities
        if not self.system["use_capacities_existing"]:
            return df_output
        f_name, scenario_factor = self.scenario_dict.get_param_file(self.element.name, file_name)
        if f"{f_name}.csv" in os.listdir(self.folder_path):
            df_input = self.read_input_csv(f_name)
            # fill output dataframe
            df_output = self.extract_general_input_data(df_input, df_output, "year_construction", index_name_list, default_value=0, time_steps=None)
            # get reference year
            reference_year = self.system["reference_year"]
            # calculate remaining lifetime
            df_output[df_output > 0] = - reference_year + df_output[df_output > 0] + self.element.lifetime[0]
        # apply scenario factor
        return df_output*scenario_factor

    def extract_pwa_capex(self):
        """ reads input data and restructures the dataframe to return (multi)indexed dict

        :return pwa_dict: dictionary with pwa parameters """
        attribute_name = "capex_specific_conversion"
        index_sets = ["set_nodes", "set_time_steps_yearly"]
        time_steps = "set_time_steps_yearly"
        unit_category = {"money": 1, "energy_quantity": -1, "time": 1}
        # import all input data
        df_input_nonlinear, has_unit_nonlinear = self.read_pwa_capex_files(file_type="nonlinear_")
        # if nonlinear
        if df_input_nonlinear is not None:
            if not has_unit_nonlinear:
                raise NotImplementedError("Nonlinear pwa files must have units")
            # select data
            pwa_dict = {}
            # extract all data values
            nonlinear_values = {}

            df_input_nonlinear["capex"] = df_input_nonlinear["capex"] * df_input_nonlinear["capacity"]
            for column in df_input_nonlinear.columns:
                nonlinear_values[column] = df_input_nonlinear[column].to_list()

            breakpoint_variable = df_input_nonlinear.columns[0]
            breakpoints = df_input_nonlinear[breakpoint_variable].to_list()

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
                             [value_variable][idx_breakpoint] for idx_breakpoint, breakpoint in enumerate(breakpoints)
                                                 if min_capacity_tech <= breakpoint <= max_capacity_tech
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
                    f"There are both linearly and nonlinearly modeled variables in capex of {self.element.name}. Not yet implemented")
        # linear
        else:
            is_pwa = False
            linear_dict = {}
            linear_dict["capex"] = self.extract_input_data(attribute_name, index_sets=index_sets,
                                                           time_steps=time_steps, unit_category=unit_category)
            return linear_dict, is_pwa

    def read_pwa_capex_files(self, file_type=str()):
        """ reads pwa files

        :param file_type: either breakpointsPWA, linear, or nonlinear
        :return df_input: raw input file"""
        df_input = self.read_input_csv(file_type + "capex")
        has_unit = False
        if df_input is not None:
            string_row = df_input.map(lambda x: pd.to_numeric(x, errors='coerce')).isna().any(axis=1)
            if string_row.any():
                unit_row = df_input.loc[string_row]
                #save non-linear capex units for consistency checks
                if file_type == "nonlinear_":
                    self.element.units_nonlinear_capex_files = {"nonlinear": unit_row}
                # elif file_type == "breakpoints_pwa_":
                #     self.element.units_nonlinear_capex_files["breakpoints"] = unit_row
                df_input = df_input.loc[~string_row]
                if isinstance(unit_row, pd.DataFrame):
                    unit_row = unit_row.squeeze()
                if isinstance(unit_row, str):
                    multiplier = self.unit_handling.get_unit_multiplier(unit_row, attribute_name="capex")
                else:
                    multiplier = unit_row.apply(lambda unit: self.unit_handling.get_unit_multiplier(unit, attribute_name="capex"))
                df_input = df_input.astype(float)*multiplier
                has_unit = True
        return df_input, has_unit

    def create_default_output(self, index_sets, unit_category, file_name=None, time_steps=None, manual_default_value=None, subelement=None):
        """ creates default output dataframe

        :param index_sets: index sets of attribute. Creates (multi)index. Corresponds to order in pe.Set/pe.Param
        :param unit_category: dict defining the dimensions of the parameter's unit
        :param file_name: name of selected file.
        :param time_steps: specific time_steps of subelement
        :param manual_default_value: if given, use manual_default_value instead of searching for default value in attributes.csv
        :param subelement: dependent element for which data is extracted
        """
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
            default_value = self.extract_attribute(default_name, unit_category, subelement=subelement)

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
        self.save_unit_of_attribute(default_name, subelement)
        return df_output, default_value, index_name_list

    def save_unit_of_attribute(self, attribute_name, subelement=None):
        """ saves the unit of an attribute, converted to the base unit
        :param attribute_name: name of selected attribute
        :param subelement: dependent element for which data is extracted
        """
        # if numerics analyzed
        if self.solver["analyze_numerics"] and attribute_name is not None:
            input_unit = self.extract_attribute(attribute_name, unit_category=None, subelement=subelement, return_unit=True)
            if subelement is not None:
                attribute_name = attribute_name + "_" + subelement
            self.unit_handling.set_base_unit_combination(input_unit=input_unit, attribute=(self.element.name, attribute_name))

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
                logging.warning(f"DeprecationWarning: Generic time indices (used in {file_name}) will not be supported for input data with yearly time steps any longer! Use the corresponding years (e.g. 2022,2023,...) as time indices instead")
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
                if isinstance(values, int):
                    index = [0]
                    is_float = False
                    int_check = True
                elif isinstance(values, float):
                    index = [0]
                    is_float = True
                    int_check = values.is_integer()
                else:
                    index = list(range(len(values)))
                    is_float = any(isinstance(v, float) for v in values)
                    int_check = all([float(v).is_integer() for v in values])
                # check that correct dtype of values
                if df_output.dtype == int and is_float:
                    if int_check:
                        if isinstance(values, list):
                            values = [int(v) for v in values]
                        else:
                            values = int(values)
                    else:
                        raise ValueError(f"Values in {column} are not integers, but should be")
                df_output.loc[location, index] = values
        return df_output
