"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Class is defining to read in the results of an Optimization problem.
==========================================================================================================================================================================="""
import logging

import h5py
import numpy as np
import pandas as pd
import importlib
import json
import zlib
import os

from zen_garden import utils
from zen_garden.model.objects.time_steps import SequenceTimeStepsDicts


class Results(object):
    """
    This class reads in the results after the pipeline has run
    """

    def __init__(self, path, scenarios=None, load_opt=False):
        """
        Initializes the Results class with a given path
        :param path: Path to the output of the optimization problem
        :param scenarios: A list of scenarios to load, defaults to all scenarios
        :param load_opt: Optionally load the opt dictionary as well
        """

        # get the abs path
        self.path = os.path.abspath(path)

        # check if the path exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"No such file or directory: {self.path}")

        # load the onetime stuff
        self.results = {}
        self.results["analysis"] = self.load_analysis(self.path)
        self.results["scenarios"] = self.load_scenarios(self.path)
        self.results["solver"] = self.load_solver(self.path)
        self.results["system"] = self.load_system(self.path)

        # get the years
        self.years = list(range(0, self.results["system"]["optimized_years"]))

        # this is a list for lazy dict to keep the datasets open
        self._lazydicts = []

        # if we only want to load a subset
        if scenarios is not None:
            self.has_scenarios = True
            self.scenarios = []
            # append prefix if necessary
            for scenario in scenarios:
                if not scenario.startswith("scenario_"):
                    self.scenarios.append(f"scenario_{scenario}")
                else:
                    self.scenarios.append(scenario)
        # we have scenarios and load all
        elif self.results["system"]["conduct_scenario_analysis"]:
            self.has_scenarios = True
            self.scenarios = [f"scenario_{scenario}" for scenario in self.results["scenarios"].keys()]
        # there are no scenarios
        else:
            self.has_scenarios = False
            self.scenarios = [None]
        # myopic foresight
        if self.results["system"]["use_rolling_horizon"]:
            self.has_MF = True
            self.mf = [f"MF_{step_horizon}" for step_horizon in self.years]
        else:
            self.has_MF = False
            self.mf = [None]

        # cycle through the dirs
        for scenario in self.scenarios:
            # init dict
            self.results[scenario] = {}

            # load the corresponding timestep dict
            time_dict = self.load_sequence_time_steps(self.path, scenario)
            self.results[scenario]["dict_sequence_time_steps"] = time_dict
            self.results[scenario]["sequence_time_steps_dicts"] = SequenceTimeStepsDicts(time_dict)

            for mf in self.mf:
                # init dict
                self.results[scenario][mf] = {}

                # get the current path
                subfolder = ""
                if self.has_scenarios:
                    # handle scenarios
                    subfolder += scenario
                    # add the buffer if necessary
                    if self.has_MF:
                        subfolder += "_"
                # deal with MF
                if self.has_MF:
                    subfolder += mf

                # Add together
                current_path = os.path.join(self.path, subfolder)

                # create dict containing params and vars
                self.results[scenario][mf]["pars_and_vars"] = {}
                pars = self.load_params(current_path, lazy=True)
                self._lazydicts.append([pars])
                self.results[scenario][mf]["pars_and_vars"].update(pars)
                vars = self.load_vars(current_path, lazy=True)
                self._lazydicts.append([vars])
                self.results[scenario][mf]["pars_and_vars"].update(vars)

                # the opt we only load when requested
                if load_opt:
                    self.results[scenario][mf]["optdict"] = self.load_opt(current_path)

        # load the time step duration, these are normal dataframe calls (dicts in case of scenarios)
        self.time_step_operational_duration = self.load_time_step_operation_duration()
        self.time_step_storage_duration = self.load_time_step_storage_duration()

    @classmethod
    def _read_file(cls, name, lazy=True):
        """
        Reads out a file and decompresses it if necessary
        :param name: File name without extension
        :param lazy: When possible, load lazy
        :return: The decompressed content of the file as dict like object
        """

        # h5 version
        if os.path.exists(f"{name}.h5"):
            content = utils.load(f"{name}.h5")
            if not lazy:
                content = content.unlazy(return_dict=True)

            return content

        # compressed version
        if os.path.exists(f"{name}.gzip"):
            with open(f"{name}.gzip", "rb") as f:
                content_compressed = f.read()
            content = zlib.decompress(content_compressed).decode()
            return json.loads(content)

        # normal version
        if os.path.exists(f"{name}.json"):
            with open(f"{name}.json", "r") as f:
                content = f.read()
            return json.loads(content)

        # raise Error if nothing is found
        raise FileNotFoundError(f"The file does not exists as json or gzip: {name}")

    @classmethod
    def _dict2df(cls, dict_raw):
        """
        Transforms a parameter or variable dict to a dict containing actual pandas dataframes and not serialized jsons
        :param dict_raw: The raw dict to parse
        :return: A dict containing actual dataframes in the dataframe keys
        """

        # transform back to dataframes
        dict_df = dict()
        for key, value in dict_raw.items():
            # init the dict for the variable
            dict_df[key] = dict()

            # the docstring we keep
            dict_df[key]['docstring'] = dict_raw[key]['docstring']

            # unpack if necessary
            if isinstance(dict_raw[key]['dataframe'], np.ndarray):
                json_dump = zlib.decompress(dict_raw[key]['dataframe']).decode()
            else:
                json_dump = json.dumps(dict_raw[key]['dataframe'])

            # the dataframe we transform to an actual dataframe
            dict_df[key]['dataframe'] = pd.read_json(json_dump, orient="table")

        return dict_df

    @classmethod
    def _to_df(cls, string):
        """
        Transforms a parameter or variable dataframe (compressed) string into an actual pandas dataframe
        :param string: The string to decode
        :return: The corresponding datafram
        """

        # transform back to dataframes
        if isinstance(string, np.ndarray):
            json_dump = zlib.decompress(string).decode()
        else:
            json_dump = json.dumps(string)

        return pd.read_json(json_dump, orient="table")

    @classmethod
    def load_params(cls, path, lazy=False):
        """
        Loads the parameter dict from a given path
        :param path: Path to load the parameter dict from
        :param lazy: Load lazy, this will not transform the data into dataframes
        :return: The parameter dict
        """

        # load the raw dict
        raw_dict = cls._read_file(os.path.join(path, "param_dict"))

        if lazy:
            return raw_dict
        else:
            return cls._dict2df(raw_dict)

    @classmethod
    def load_vars(cls, path, lazy=False):
        """
        Loads the var dict from a given path
        :param path: Path to load the var dict from
        :param lazy: Load lazy, this will not transform the data into dataframes
        :return: The var dict
        """

        # load the raw dict
        raw_dict = cls._read_file(os.path.join(path, "var_dict"))

        if lazy:
            return raw_dict
        else:
            return cls._dict2df(raw_dict)

    @classmethod
    def load_system(cls, path):
        """
        Loads the system dict from a given path
        :param path: Directory to load the dictionary from
        :return: The system dictionary
        """

        # get the dict
        raw_dict = cls._read_file(os.path.join(path, "system"))

        return raw_dict

    @classmethod
    def load_analysis(cls, path):
        """
        Loads the analysis dict from a given path
        :param path: Directory to load the dictionary from
        :return: The analysis dictionary
        """

        # get the dict
        raw_dict = cls._read_file(os.path.join(path, "analysis"))

        return raw_dict

    @classmethod
    def load_solver(cls, path):
        """
        Loads the solver dict from a given path
        :param path: Directory to load the dictionary from
        :return: The analysis dictionary
        """

        # get the dict
        raw_dict = cls._read_file(os.path.join(path, "solver"))

        return raw_dict

    @classmethod
    def load_scenarios(cls, path):
        """
        Loads the scenarios dict from a given path
        :param path: Directory to load the dictionary from
        :return: The analysis dictionary
        """

        # get the dict
        raw_dict = cls._read_file(os.path.join(path, "scenarios"))

        return raw_dict

    @classmethod
    def load_opt(cls, path):
        """
        Loads the opt dict from a given path
        :param path: Directory to load the dictionary from
        :return: The analysis dictionary
        """

        # get the dict
        raw_dict = cls._read_file(os.path.join(path, "opt_dict"))

        return raw_dict

    @classmethod
    def load_sequence_time_steps(cls, path, scenario=None):
        """
        Loads the dict_sequence_time_steps from a given path
        :param path: Path to load the dict from
        :param scenario: Name of the scenario to load
        :return: dict_sequence_time_steps
        """
        # get the file name
        fname = os.path.join(path, "dict_all_sequence_time_steps")
        if scenario is not None:
            fname += f"_{scenario}"

        # get the dict
        dict_sequence_time_steps = cls._read_file(fname, lazy=False)

        # tranform all lists to arrays
        return cls.expand_dict(dict_sequence_time_steps)

    @classmethod
    def expand_dict(cls, dictionary):
        """
        Creates a copy of the dictionary where all lists are recursively transformed to numpy arrays
        :param dictionary: The input dictionary
        :return: A copy of the dictionary containing arrays instead of lists
        """
        # create a copy of the dict to avoid overwrite
        out_dict = dict()

        # faltten all arrays
        for k, v in dictionary.items():
            # transform 'null' keys to None
            if k == 'null':
                k = None

            # recursive call
            if isinstance(v, (dict, utils.LazyHdfDict)):
                out_dict[k] = cls.expand_dict(v)  # flatten the array to list
            elif isinstance(v, list):
                # Note: list(v) creates a list of np objects v.tolist() not
                out_dict[k] = np.array(v)
            # take as is
            else:
                out_dict[k] = v

        return out_dict

    def get_df(self, name, is_storage=False, scenario=None, to_csv=None, csv_kwargs=None):
        """
        Extracts the dataframe from the results
        :param name: The name of the dataframe to extract
        :param is_storage: Whether it is a storage or not
        :param scenario: If multiple scenarios are in the results, only consider this one
        :param to_csv: Save the dataframes to a csv file
        :param csv_kwargs: additional keyword arguments forwarded to the to_csv method of pandas
        :return: The dataframe that should have been extracted. If multiple scenarios are present a dictionary
                 with scenarios as keys and dataframes as value is returned
        """

        # select the scenarios
        if scenario is not None:
            scenarios = [scenario]
        else:
            scenarios = self.scenarios

        # loop
        _data = {}
        for scenario in scenarios:
            # we get the timestep dict
            sequence_time_steps_dicts = self.results[scenario]["sequence_time_steps_dicts"]

            if not self.has_MF:
                # we set the dataframe of the variable into the data dict
                _data[scenario] = self._to_df(self.results[scenario][None]["pars_and_vars"][name]["dataframe"])

            else:
                # init the scenario
                _mf_data = {}

                # cycle through all MFs
                for year, mf in enumerate(self.mf):
                    _var = self._to_df(self.results[scenario][mf]["pars_and_vars"][name]["dataframe"])

                    # single element that is not a year
                    if len(_var) == 1 and len(_var.index[0]) == 1 and not np.isfinite(_var.index[0]):
                        _data[scenario] = _var
                        break
                    # if the year is in the index (no multiindex)
                    elif year in _var.index:
                        _mf_data[year] = _var.loc[year]
                        yearly_component = True
                    else:
                        # unstack the year
                        _varSeries = _var.squeeze().unstack()
                        # get type of time steps
                        ts_type = self._get_ts_type(_varSeries,name,force_output=True)
                        # if all columns in years (drop the value level)
                        # if _varSeries.columns.droplevel(0).difference(self.years).empty:
                        if ts_type == "yearly":
                            # get the data
                            tmp_data = _varSeries[year]
                            # rename
                            tmp_data.name = _varSeries.columns.name
                            # set
                            _mf_data[year] = tmp_data
                            yearly_component = True
                            time_header = _varSeries.columns.name
                        # if more time steps than years, then it is operational ts (we drop value in columns)
                        # elif pd.to_numeric(_varSeries.columns.droplevel(0), errors="coerce").equals(_varSeries.columns.droplevel(0)):
                        elif ts_type is not None:
                            if ts_type == "storage":
                                techProxy = [k for k in self.results[scenario]["dict_sequence_time_steps"]["operation"].keys() if "storage_level" in k.lower()][0]
                            else:
                                techProxy = [k for k in self.results[scenario]["dict_sequence_time_steps"]["operation"].keys() if "storage_level" not in k.lower()][0]
                            # get the timesteps
                            time_steps_year = sequence_time_steps_dicts.encode_time_step(techProxy, sequence_time_steps_dicts.decode_time_step(None, year, "yearly"), yearly=True)
                            # get the data
                            tmp_data = _varSeries[[tstep for tstep in time_steps_year]]
                            # rename
                            tmp_data.name = _varSeries.columns.name
                            # set
                            _mf_data[year] = tmp_data
                            yearly_component = False
                            time_header = _varSeries.columns.name
                        # else not a time index
                        else:
                            _data[scenario] = _varSeries.stack()
                            break
                # This is a for-else, it is triggered if we did not break the loop
                else:
                    # deal with the years
                    if yearly_component:
                        # concat
                        new_header = [time_header]+tmp_data.index.names
                        new_order = tmp_data.index.names + [time_header]
                        _df = pd.concat(_mf_data, axis=0, keys=_mf_data.keys(),names=new_header).reorder_levels(new_order)
                        _dfIndex = _df.index.copy()
                        for level, codes in enumerate(_df.index.codes):
                            if len(np.unique(codes)) == 1 and np.unique(codes) == 0:
                                _dfIndex = _dfIndex.droplevel(level)
                                break
                        _df.index = _dfIndex
                        # if year not in _df.index:
                        #     _indexSort = list(range(0, _df.index.nlevels))
                        #     _indexSort.append(_indexSort[0])
                        #     _indexSort.pop(0)
                        #     _df = _df.reorder_levels(_indexSort)
                    else:
                        _df = pd.concat(_mf_data, axis=1)
                        _df.columns = _df.columns.droplevel(0)
                        _df = _df.sort_index(axis=1).stack()

                    _data[scenario] = _df

        # transform all dataframes to pd.Series with the element_name as name
        for k, v in _data.items():
            if not isinstance(v, pd.Series):
                # to series
                series = pd.Series(data=v["value"], index=v.index)
                series.name = name
                # set
                _data[k] = series
            # we just make sure the name is right
            else:
                v.name = name
                _data[k] = v

        # get the path to the csv file
        if to_csv is not None:
            fname, _ = os.path.splitext(to_csv)

            # deal with additional args
            if csv_kwargs is None:
                csv_kwargs = {}

        # if we only had a single scenario no need for the wrapper
        if len(scenarios) == 1:
            # save if necessary
            if to_csv is not None:
                _data[scenario].to_csv(f"{fname}.csv", **csv_kwargs)
            return _data[scenario]

        # return the dict
        else:
            # save if necessary
            if to_csv is not None:
                for scenario in scenarios:
                    _data[scenario].to_csv(f"{fname}_{scenario}.csv", **csv_kwargs)
            return _data

    def load_time_step_operation_duration(self):
        """
        Loads duration of operational time steps
        """
        return self.get_df("time_steps_operation_duration")

    def load_time_step_storage_duration(self):
        """
        Loads duration of operational time steps
        """
        return self.get_df("time_steps_storage_level_duration", is_storage=True)

    def get_full_ts(self, component, element_name=None, year=None, scenario=None):
        """
        Calculates the full timeseries for a given element
        :param component: Either the dataframe of a component as pandas.Series or the name of the component
        :param element_name: The name of the element
        :param scenario: The scenario for with the component should be extracted (only if needed)
        :return: A dataframe containing the full timeseries of the element
        """
        # extract the data
        component_name, component_data = self._get_component_data(component, scenario)
        # timestep dict
        sequence_time_steps_dicts = self.results[scenario]["sequence_time_steps_dicts"]

        ts_type = self._get_ts_type(component_data, component_name)

        if ts_type == "yearly":
            if element_name is not None:
                component_data = component_data.loc[element_name]
            # component indexed by yearly component
            if year is not None:
                if year in component_data.columns:
                    return component_data[year]
                else:
                    print(f"WARNING: year {year} not in years {component_data.columns}. Return component values for all years")
                    return component_data
            else:
                return component_data
        elif ts_type == "operational":
            _storage_string = ""
        else:
            _storage_string = "_storage_level"

        # calculate the full time series
        _output_temp = {}
        for row in component_data.index:
            # we know the name
            if element_name:
                _sequence_time_steps = sequence_time_steps_dicts.get_sequence_time_steps(element_name + _storage_string)
            # we extract the name
            else:
                _sequence_time_steps = sequence_time_steps_dicts.get_sequence_time_steps(row[0] + _storage_string)

            # throw together
            _sequence_time_steps = _sequence_time_steps[np.in1d(_sequence_time_steps, list(component_data.columns))]
            _output_temp[row] = component_data.loc[row, _sequence_time_steps].reset_index(drop=True)
            if year is not None:
                if year in self.years:
                    hours_of_year = self._get_hours_of_year(year)
                    _output_temp[row] = (_output_temp[row][hours_of_year]).reset_index(drop=True)
                else:
                    print(f"WARNING: year {year} not in years {self.years}. Return component values for all years")

        # concat and return
        outputDf = pd.concat(_output_temp, axis=0, keys=_output_temp.keys()).unstack()
        return outputDf

    def get_total(self, component, element_name=None, year=None, scenario=None, split_years=True):
        """
        Calculates the total Value of a component
        :param component: Either a dataframe as returned from <get_df> or the name of the component
        :param element_name: The element name to calculate the value for, defaults to all elements
        :param year: The year to calculate the value for, defaults to all years
        :param scenario: The scenario to calculate the total value for
        :param split_years: Calculate the value for each year individually
        :return: A dataframe containing the total value with the specified paramters
        """
        # extract the data
        component_name, component_data = self._get_component_data(component, scenario)
        # timestep dict
        sequence_time_steps_dicts = self.results[scenario]["sequence_time_steps_dicts"]

        ts_type = self._get_ts_type(component_data, component_name)

        if ts_type == "yearly":
            if element_name is not None:
                component_data = component_data.loc[element_name]
            if year is not None:
                if year in component_data.columns:
                    return component_data[year]
                else:
                    print(f"WARNING: year {year} not in years {component_data.columns}. Return total value for all years")
                    return component_data.sum(axis=1)
            else:
                if split_years:
                    return component_data
                else:
                    return component_data.sum(axis=1)
        elif ts_type == "operational":
            _isStorage = False
            _storage_string = ""
        else:
            _isStorage = True
            _storage_string = "_storage_level"

        # extract time step duration
        time_step_duration = self._get_ts_duration(scenario, is_storage=_isStorage)

        # If we have an element name
        if element_name is not None:
            # check that it is in the index
            assert element_name in component_data.index.get_level_values(level=0), f"element {element_name} is not found in index of {component_name}"
            # get the index
            component_data = component_data.loc[element_name]
            time_step_duration_element = time_step_duration.loc[element_name]

            if year is not None:
                # only for the given year
                time_steps_year = sequence_time_steps_dicts.encode_time_step(element_name + _storage_string, sequence_time_steps_dicts.decode_time_step(None, year, "yearly"), yearly=True)
                total_value = (component_data * time_step_duration_element)[time_steps_year].sum(axis=1)
            else:
                # for all years
                if split_years:
                    total_value_temp = pd.DataFrame(index=component_data.index, columns=self.years)
                    for year_temp in self.years:
                        # set a proxy for the element name
                        time_steps_year = sequence_time_steps_dicts.encode_time_step(element_name + _storage_string, sequence_time_steps_dicts.decode_time_step(None, year_temp, "yearly"), yearly=True)
                        total_value_temp[year_temp] = (component_data * time_step_duration_element)[time_steps_year].sum(axis=1)
                    total_value = total_value_temp
                else:
                    total_value = (component_data * time_step_duration_element).sum(axis=1)

        # if we do not have an element name
        else:
            total_value = component_data.apply(lambda row: row * time_step_duration.loc[row.name[0]], axis=1)
            if year is not None:
                # set a proxy for the element name
                element_name_proxy = component_data.index.get_level_values(level=0)[0]
                time_steps_year = sequence_time_steps_dicts.encode_time_step(element_name_proxy + _storage_string, sequence_time_steps_dicts.decode_time_step(None, year, "yearly"), yearly=True)
                total_value = total_value[time_steps_year].sum(axis=1)
            else:
                if split_years:
                    total_value_temp = pd.DataFrame(index=total_value.index, columns=self.years)
                    for year_temp in self.years:
                        # set a proxy for the element name
                        element_name_proxy = component_data.index.get_level_values(level=0)[0]
                        time_steps_year = sequence_time_steps_dicts.encode_time_step(element_name_proxy + _storage_string, sequence_time_steps_dicts.decode_time_step(None, year_temp, "yearly"),
                                                                                     yearly=True)
                        total_value_temp[year_temp] = total_value[time_steps_year].sum(axis=1)
                    total_value = total_value_temp
                else:
                    total_value = total_value.sum(axis=1)
        return total_value

    def _get_ts_duration(self, scenario=None, is_storage=False):
        """ extracts the time steps duration """
        # extract the right timestep duration
        if self.has_scenarios:
            if scenario is None:
                raise ValueError("Please specify a scenario!")
            else:
                if is_storage:
                    time_step_duration = self.time_step_storage_duration[scenario].unstack()
                else:
                    time_step_duration = self.time_step_operational_duration[scenario].unstack()
        else:
            if is_storage:
                time_step_duration = self.time_step_storage_duration.unstack()
            else:
                time_step_duration = self.time_step_operational_duration.unstack()
        return time_step_duration

    def _get_component_data(self, component, scenario=None):
        """ extracts the data for a component"""
        # extract the data
        if isinstance(component, str):
            component_name = component
            # only use the data from one scenario if specified
            if scenario is not None:
                component_data = self.get_df(component)[scenario].unstack()
            else:
                component_data = self.get_df(component).unstack()
        elif isinstance(component, pd.Series):
            component_name = component.name
            component_data = component.unstack()
        else:
            raise TypeError(f"Type {type(component).__name__} of input is not supported.")

        return component_name, component_data

    def _get_ts_type(self, component_data, component_name,force_output = False):
        """ get time step type (operational, storage, yearly) """
        _header_operational = self.results["analysis"]["header_data_inputs"]["set_time_steps_operation"]
        _header_storage = self.results["analysis"]["header_data_inputs"]["set_time_steps_storage_level"]
        _header_yearly = self.results["analysis"]["header_data_inputs"]["set_time_steps_yearly"]
        if component_data.columns.name == _header_operational:
            return "operational"
        elif component_data.columns.name == _header_storage:
            return "storage"
        elif component_data.columns.name == _header_yearly:
            return "yearly"
        else:
            if force_output:
                return None
            else:
                raise KeyError(f"Column index name of '{component_name}' ({component_data.columns.name}) is unknown. Should be (operational, storage, yearly)")

    def _get_hours_of_year(self, year):
        """ get total hours of year """
        _total_hours_per_year = self.results["system"]["unaggregated_time_steps_per_year"]
        _hours_of_year = list(range(year * _total_hours_per_year, (year + 1) * _total_hours_per_year))
        return _hours_of_year

    def __str__(self):
        return f"Results of '{self.path}'"


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("module", "config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.config

    model_name = os.path.basename(config.analysis["dataset"])
    if os.path.exists(out_folder := os.path.join(config.analysis["folder_output"], model_name)):
        r = Results(out_folder)
    else:
        logging.critical("No results folder found!")
