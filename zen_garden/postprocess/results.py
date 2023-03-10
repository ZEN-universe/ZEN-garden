"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Class is defining to read in the results of an Optimization problem.
==========================================================================================================================================================================="""
import logging
import warnings

import h5py
import numpy as np
import pandas as pd
import importlib
import json
import zlib
import os
import matplotlib.pyplot as plt

from zen_garden import utils
from zen_garden.model.objects.time_steps import TimeStepsDicts


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
            self.results[scenario]["sequence_time_steps_dicts"] = TimeStepsDicts(time_dict)
            # load the operation2year and year2operation time steps
            for element in time_dict["operation"]:
                self.results[scenario]["sequence_time_steps_dicts"].set_time_steps_operation2year_both_dir(element,time_dict["operation"][element],time_dict["yearly"][None])
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

                #create dict containing sets
                self.results[scenario][mf]["sets"] = {}
                sets = self.load_sets(current_path, lazy=True)
                self._lazydicts.append([sets])
                self.results[scenario][mf]["sets"].update(sets)

                # create dict containing params and vars
                self.results[scenario][mf]["pars_and_vars"] = {}
                pars = self.load_params(current_path, lazy=True)
                self._lazydicts.append([pars])
                self.results[scenario][mf]["pars_and_vars"].update(pars)
                vars = self.load_vars(current_path, lazy=True)
                self._lazydicts.append([vars])
                self.results[scenario][mf]["pars_and_vars"].update(vars)

                # load duals
                if self.results["solver"]["add_duals"]:
                    duals = self.load_duals(current_path, lazy = True)
                    self.results[scenario][mf]["duals"] = duals
                # the opt we only load when requested
                if load_opt:
                    self.results[scenario][mf]["optdict"] = self.load_opt(current_path)

        # load the time step duration, these are normal dataframe calls (dicts in case of scenarios)
        self.time_step_operational_duration = self.load_time_step_operation_duration()
        self.time_step_storage_duration = self.load_time_step_storage_duration()

        data_1 = self.get_df("output_flow")
        data = self.get_df("set_reference_carriers", is_set=True)
        #self.plot("input_flow", node_edit=True)
        #self.standard_plots()
        self.plot_energy_balance("DE", "heat", 2022)
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
        :return: The corresponding dataframe
        """

        # transform back to dataframes
        if isinstance(string, np.ndarray):
            json_dump = zlib.decompress(string).decode()
        else:
            json_dump = json.dumps(string)

        return pd.read_json(json_dump, orient="table")

    @classmethod
    def load_sets(cls, path, lazy=False):
        """
        Loads the set dict from a given path
        :param path: Path to load the parameter dict from
        :param lazy: Load lazy, this will not transform the data into dataframes
        :return: The set dict
        """

        # load the raw dict
        raw_dict = cls._read_file(os.path.join(path, "set_dict"))

        if lazy:
            return raw_dict
        else:
            return cls._dict2df(raw_dict)

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
    def load_duals(cls, path, lazy=False):
        """
        Loads the dual dict from a given path
        :param path: Path to load the dual dict from
        :param lazy: Load lazy, this will not transform the data into dataframes
        :return: The var dict
        """

        # load the raw dict
        raw_dict = cls._read_file(os.path.join(path, "dual_dict"))

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

    def get_df(self, name, scenario=None, to_csv=None, csv_kwargs=None,is_dual=False, is_set=False):
        """
        Extracts the dataframe from the results
        :param name: The name of the dataframe to extract
        :param scenario: If multiple scenarios are in the results, only consider this one
        :param to_csv: Save the dataframes to a csv file
        :param csv_kwargs: additional keyword arguments forwarded to the to_csv method of pandas
        :param is_dual: if dual variable dict is selected
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
                if not is_set:
                    # we set the dataframe of the variable into the data dict
                    if not is_dual:
                        _data[scenario] = self._to_df(self.results[scenario][None]["pars_and_vars"][name]["dataframe"])
                    else:
                        _data[scenario] = self._to_df(self.results[scenario][None]["duals"][name]["dataframe"])
                else:
                    _data[scenario] = self._to_df(self.results[scenario][None]["sets"][name]["dataframe"])

            else:
                # init the scenario
                _mf_data = {}
                _is_multiindex = False
                # cycle through all MFs
                for year, mf in enumerate(self.mf):
                    if not is_dual:
                        _var = self._to_df(self.results[scenario][mf]["pars_and_vars"][name]["dataframe"])
                    else:
                        _var = self._to_df(self.results[scenario][mf]["duals"][name]["dataframe"])
                    # # single element that is not a year
                    # if len(_var) == 1 and _var.index.nlevels == 1 and not np.isfinite(_var.index[0]):
                    #     _data[scenario] = _var
                    #     break
                    # # if the year is in the index (no multiindex)
                    # elif year in _var.index:
                    #     _mf_data[year] = _var.loc[year]
                    #     yearly_component = True

                    # no multiindex
                    if _var.index.nlevels == 1:
                        ts_type = self._get_ts_type(_var.T,name,force_output=True)
                        # if yearly variable
                        if ts_type == "yearly":
                            _mf_data[year] = _var.loc[year].squeeze()
                            yearly_component = True
                            time_header = _var.index.name
                        elif ts_type is None:
                            _data[scenario] = _var
                            break
                        else:
                            raise KeyError(f"The time step type '{ts_type}' was not expected for variable '{name}'")
                    # multiindex
                    else:
                        _is_multiindex = True
                        # unstack the year
                        _varSeries = _var["value"].unstack()
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
                            time_steps_year = sequence_time_steps_dicts.get_time_steps_year2operation(techProxy, year)
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
                        if _is_multiindex:
                            new_header = [time_header]+tmp_data.index.names
                            new_order = tmp_data.index.names + [time_header]
                            _df = pd.concat(_mf_data, axis=0, keys=_mf_data.keys(),names=new_header).reorder_levels(new_order)
                            _df_index = _df.index.copy()
                            for level, codes in enumerate(_df.index.codes):
                                if len(np.unique(codes)) == 1 and np.unique(codes) == 0:
                                    _df_index = _df_index.droplevel(level)
                                    break
                            _df.index = _df_index
                        else:
                            _df = pd.Series(_mf_data,index=_mf_data.keys())
                            _df.index.name = time_header

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
        return self.get_df("time_steps_storage_level_duration")

    def get_full_ts(self, component, element_name=None, year=None, scenario=None,is_dual = False):
        """
        Calculates the full timeseries for a given element
        :param component: Either the dataframe of a component as pandas.Series or the name of the component
        :param element_name: The name of the element
        :param scenario: The scenario for with the component should be extracted (only if needed)
        :return: A dataframe containing the full timeseries of the element
        """
        # extract the data
        component_name, component_data = self._get_component_data(component, scenario,is_dual = is_dual)

        # timestep dict
        sequence_time_steps_dicts = self.results[scenario]["sequence_time_steps_dicts"]
        ts_type = self._get_ts_type(component_data, component_name)
        if is_dual:
            annuity = self._get_annuity()
        else:
            annuity = pd.Series(index=self.years,data=1)
        if isinstance(component_data,pd.Series):
            return component_data/annuity
        if ts_type == "yearly":
            if element_name is not None:
                component_data = component_data.loc[element_name]
            component_data = component_data.div(annuity,axis=1)
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
            time_step_duration = self._get_ts_duration(scenario, is_storage=False)
        else:
            _storage_string = "_storage_level"
            time_step_duration = self._get_ts_duration(scenario, is_storage=True)
        if element_name is not None:
            component_data = component_data.loc[element_name]
        # calculate the full time series
        _output_temp = {}
        # extract time step duration
        for row in component_data.index:
            # we know the name
            if element_name:
                _sequence_time_steps = sequence_time_steps_dicts.get_sequence_time_steps(element_name + _storage_string)
                ts_duration = time_step_duration.loc[element_name]
            # we extract the name
            else:
                _sequence_time_steps = sequence_time_steps_dicts.get_sequence_time_steps(row[0] + _storage_string)
                ts_duration = time_step_duration.loc[row[0]]
            # if dual variables, divide by time step operational duration
            if is_dual:

                component_data.loc[row] = component_data.loc[row]/ts_duration
                for _year in annuity.index:
                    if element_name:
                        _yearly_ts = sequence_time_steps_dicts.get_time_steps_year2operation(element_name + _storage_string,_year)
                    else:
                        _yearly_ts = sequence_time_steps_dicts.get_time_steps_year2operation(row[0] + _storage_string,_year)
                    component_data.loc[row,_yearly_ts] = component_data.loc[row,_yearly_ts] / annuity[_year]

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
        output_df = pd.concat(_output_temp, axis=0, keys=component_data.index).unstack()
        return output_df

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
        if isinstance(component_data,pd.Series):
            return component_data
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
                time_steps_year = sequence_time_steps_dicts.get_time_steps_year2operation(element_name + _storage_string,  year)
                total_value = (component_data * time_step_duration_element)[time_steps_year].sum(axis=1)
            else:
                # for all years
                if split_years:
                    total_value_temp = pd.DataFrame(index=component_data.index, columns=self.years)
                    for year_temp in self.years:
                        # set a proxy for the element name
                        time_steps_year = sequence_time_steps_dicts.get_time_steps_year2operation(element_name + _storage_string, year_temp)
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
                time_steps_year = sequence_time_steps_dicts.get_time_steps_year2operation(element_name_proxy + _storage_string, year)
                total_value = total_value[time_steps_year].sum(axis=1)
            else:
                if split_years:
                    total_value_temp = pd.DataFrame(index=total_value.index, columns=self.years)
                    for year_temp in self.years:
                        # set a proxy for the element name
                        element_name_proxy = component_data.index.get_level_values(level=0)[0]
                        time_steps_year = sequence_time_steps_dicts.get_time_steps_year2operation(element_name_proxy + _storage_string, year_temp)
                        total_value_temp[year_temp] = total_value[time_steps_year].sum(axis=1)
                    total_value = total_value_temp
                else:
                    total_value = total_value.sum(axis=1)
        return total_value

    def get_dual(self,constraint,scenario=None, element_name=None, year=None):
        """ extracts the dual variables of a constraint """
        if not self.results["solver"]["add_duals"]:
            logging.warning("Duals are not calculated. Skip.")
            return
        _duals = self.get_full_ts(component=constraint,scenario=scenario,is_dual=True, element_name=element_name, year=year)
        return _duals

    def _get_annuity(self):
        """ discounts the duals """
        system = self.results["system"]
        # calculate annuity
        discount_rate = self.results["analysis"]["discount_rate"]
        annuity = pd.Series(index=self.years,dtype=float)
        for year in self.years:
            if year == self.years[-1]:
                interval_between_years = 1
            else:
                interval_between_years = system["interval_between_years"]
            if self.has_MF:
                annuity[year] = sum(((1 / (1 + discount_rate)) ** (_intermediate_time_step)) for _intermediate_time_step in range(0, interval_between_years))
            else:
                annuity[year] = sum(((1 / (1 + discount_rate)) ** (interval_between_years * (year - self.years[0]) + _intermediate_time_step))
                        for _intermediate_time_step in range(0, interval_between_years))
        return annuity

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

    def _get_component_data(self, component, scenario=None,is_dual=False):
        """ extracts the data for a component"""
        # extract the data
        if isinstance(component, str):
            component_name = component
            # only use the data from one scenario if specified
            if scenario is not None:
                component_data = self.get_df(component,is_dual=is_dual)[scenario]
            else:
                component_data = self.get_df(component,is_dual=is_dual)
            if isinstance(component_data.index,pd.MultiIndex):
                component_data = component_data.unstack()
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
        if isinstance(component_data,pd.Series):
            axis_name = component_data.index.name
        else:
            axis_name = component_data.columns.name
        if axis_name == _header_operational:
            return "operational"
        elif axis_name == _header_storage:
            return "storage"
        elif axis_name == _header_yearly:
            return "yearly"
        else:
            if force_output:
                return None
            else:
                raise KeyError(f"Axis index name of '{component_name}' ({axis_name}) is unknown. Should be (operational, storage, yearly)")

    def _get_hours_of_year(self, year):
        """ get total hours of year """
        _total_hours_per_year = self.results["system"]["unaggregated_time_steps_per_year"]
        _hours_of_year = list(range(year * _total_hours_per_year, (year + 1) * _total_hours_per_year))
        return _hours_of_year

    def __str__(self):
        return f"Results of '{self.path}'"

    def standard_plots(self):
        """
        Plots standard variables such as capacity, built capacity, input/output flow, total capex/opex
        """
        self.plot("capacity", yearly=True, plot_type="stacked_bar", technology_type="conversion", reference_carrier="electricity", plot_strings={"title": "Capacities of Electricity Generating Conversion Technologies", "ylabel": "Capacity [GW]"})
        self.plot("capacity", yearly=True, plot_type="stacked_bar", technology_type="conversion", reference_carrier="heat", plot_strings={"title": "Capacities of Heat Generating Conversion Technologies", "ylabel": "Capacity [GW]"})
        ##self.plot("capacity", yearly=True, plot_type="stacked_bar", technology_type="transport", plot_strings={"title": "Capacities of Transport Technologies", "ylabel": "Capacity [GW]"})
        ##self.plot("capacity", yearly=True, plot_type="stacked_bar", technology_type="storage_energy", plot_strings={"title": "Storage Capacities of Storage Technologies", "ylabel": "Capacity [GWh]"})
        ##self.plot("capacity", yearly=True, plot_type="stacked_bar", technology_type="storage_power", plot_strings={"title": "Charge/Discharge Capacities of Storage Technologies", "ylabel": "Capacity [GW]"})
        self.plot("built_capacity", yearly=True, plot_type="stacked_bar", technology_type="conversion", reference_carrier="electricity", plot_strings={"title": "Built Capacities of Electricity Generating Conversion Technologies", "ylabel": "Capacity [GW]"})
        self.plot("built_capacity", yearly=True, plot_type="stacked_bar", technology_type="conversion", reference_carrier="heat", plot_strings={"title": "Built Capacities of Heat Generating Conversion Technologies", "ylabel": "Capacity [GW]"})
        self.plot("input_flow", yearly=True, plot_type="stacked_bar", sum_technologies=True, plot_strings={"title": "Input Flows Summed Over Technologies", "ylabel": "Input Flow [GW]"})
        self.plot("output_flow", yearly=True, plot_type="stacked_bar", sum_technologies=True, plot_strings={"title": "Output Flows Summed Over Technologies", "ylabel": "Output Flow [GW]"})
        self.plot("capex_total",yearly=True, plot_type="stacked_bar", plot_strings={"title": "Total Capex", "ylabel": "Capex [MEUR]"})
        self.plot("opex_total", yearly=True, plot_type="stacked_bar", plot_strings={"title": "Total Opex", "ylabel": "Opex [MEUR]"})

    def plot_energy_balance(self, node, carrier, year, start_hour=None, duration=None):
        """
        Visualizes the energy balance of a specific carrier at a single node
        :param node: String of node of interest
        :param carrier: String of carrier of interest
        :param year: Year of interest
        :param start_hour: Specific hour of year, where plot should start (needs to be passed together with duration)
        :param duration: Number of hours that should be plotted from start_hour
        """
        plt.rcParams["figure.figsize"] = (30*1, 6.5*1)
        components = ["output_flow", "input_flow", "export_carrier_flow", "import_carrier_flow", "carrier_flow_charge", "carrier_flow_discharge", "demand_carrier"]
        lowers = ["input_flow", "import_carrier_flow", "carrier_flow_charge"]
        data_plot = pd.DataFrame()
        for component in components:
            #get full timeseries of component and extract rows of relevant node
            data_full_ts = Results.edit_nodes_v2(self.get_full_ts(component), node)

            if component in ["carrier_flow_charge", "carrier_flow_discharge"]:
                carrier = "pumped_hydro"
            #extract data of desired carrier
            data_full_ts = Results.extract_carrier(data_full_ts, carrier)
            #change sign of variables which enter the node
            if component in lowers:
                data_full_ts = data_full_ts.multiply(-1)
            #add variable name to multi-index such that they can be shown in plot legend
            data_full_ts = pd.concat([data_full_ts], keys=[component], names=["variable"])
            #drop unnecessary index levels to improve the plot legend's appearance
            if data_full_ts.index.nlevels == 3:
                data_full_ts = data_full_ts.droplevel([1,2])
            elif data_full_ts.index.nlevels == 4:
                data_full_ts = data_full_ts.droplevel([2,3])
            #transpose data frame as pandas plot function plots column-wise
            data_full_ts = data_full_ts.transpose()
            #add data of current variable to the plot data frame
            data_plot = pd.concat([data_plot, data_full_ts], axis=1)

            carrier = "electricity"

        #extract the rows of the desired year
        data_plot = data_plot.iloc[8760*(year-self.results["system"]["reference_year"]):8760*(year-self.results["system"]["reference_year"])+8760]
        #extrect specific hours of year
        if start_hour is not None and duration is not None:
            data_plot = data_plot.iloc[start_hour:start_hour+duration]
        #remove columns(technologies/variables) with constant zero value
        data_plot = data_plot.loc[:, (data_plot != 0).any(axis=0)]
        #set colors and plot data frame
        colors = plt.cm.tab20(range(data_plot.shape[1]))
        data_plot.plot(kind="area", stacked=True, color=colors, title="Energy Balance " + carrier + " " + node + " " + str(year), ylabel="Power [GW]", xlabel="Time [h]")
        plt.show()

    def plot(self, component, yearly=False, node_edit=True, sum_technologies=False, technology_type=None, plot_type=None, reference_carrier=None, plot_strings={"title": "", "ylabel": ""}):
        """
        Plots component data as specified by arguments
        :param component: Either the dataframe of a component as pandas.Series or the name of the component
        :param yearly: Operational time steps if false, else yearly time steps
        :param node_edit: If true: sum values of identical technologies and carriers over spatial distribution (nodes), if string of specific node is passed, its data is returned
        :param sum_technologies: sum values of technologies per carrier if true
        :param technology_type: specify whether transport, storage or conversion technologies should be plotted separately (useful for capacity, etc.)
        :param: plot_type: per default bar plot, passing stacked_bar will plot stacked bar plot
        :reference_carrier: specify reference carrier such as electricity, heat, etc. to extract their data
        :plot_strings: Dict of strings used to set title and labels of plot
        :return: plot
        """
        component_name, component_data = self._get_component_data(component)
        #set timeseries type
        if yearly:
            ts_type = "yearly"
        else:
            ts_type = self._get_ts_type(component_data, component_name)

        #plot variable with operational time steps
        if ts_type == "operational":
            data_full_ts = self.get_full_ts(component)
            if node_edit:
                data_full_ts = Results.edit_nodes_v2(data_full_ts, node_edit)
            plt.plot(data_full_ts.columns.values, data_full_ts.values.transpose(), lw=1/len(data_full_ts.columns.values)*3000)
            plt.xlabel("Time [hours]")
            plt.legend(data_full_ts.index.values)
            plt.ylabel("Power [GW]")

        #plot variable with yearly time steps
        elif ts_type == "yearly":
            data_total = self.get_total(component)
            if technology_type != None:
                data_total = self.extract_technology(data_total, technology_type)
            if reference_carrier != None:
                data_total = self.extract_reference_carrier(data_total, reference_carrier)
            #sum data according to chosen options
            if node_edit:
                data_total = Results.edit_nodes(data_total, node_edit)
            if sum_technologies:
                data_total = Results.sum_over_technologies(data_total)

            if plot_type == None:
                #set up bar plot
                bars = []
                for ind, row in enumerate(data_total.values):
                    bar = plt.bar(data_total.columns.values + 1/(data_total.shape[0]+1) * ind, row, 1/(data_total.shape[0]+1))
                    bars.append(bar)
                #data_total.columns.values dtype needs to be cast to an integer as edit_nodes changes the dtype to an object
                plt.xticks(np.array(data_total.columns.values, dtype=int) + 1/(data_total.shape[0]+1) * 1/2 * (data_total.shape[0]-1), np.array(data_total.columns.values, dtype=int)+self.results["system"]["reference_year"])
                plt.legend((bars),(data_total.index.values),ncols=max(1,int(data_total.shape[0]/7)))
                if component in ["input_flow","out_oput_flow"]:
                    plt.ylabel("Energy [GWh]")
                elif component in ["carbon_emissions_carrier"]:
                    plt.ylabel("Carbon Emissions [Mt]")
                elif component in ["capacity"]:
                    plt.ylabel("Capacity [GW]")
                plt.xlabel("Time [years]")
            elif plot_type == "stacked_bar":
                if isinstance(data_total, pd.Series):
                    data_total = pd.Series(np.array(data_total.values), np.array(data_total.index.values, dtype=int) + self.results["system"]["reference_year"])
                elif isinstance(data_total, pd.DataFrame):
                    data_total = data_total.transpose()
                    data_total = data_total.set_index(np.array(data_total.index.values, dtype=int) + self.results["system"]["reference_year"])
                plt.rcParams["figure.figsize"] = (9.5, 6.5)
                fig, ax = plt.subplots()
                data_total.plot(ax=ax, kind='bar', stacked=True,
                        title=plot_strings["title"], rot=0, xlabel='Year', ylabel=plot_strings["ylabel"])
                pos = ax.get_position()
                ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])
                ax.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))

        #plot variable with storage time steps
        elif ts_type == "storage":
            data_total = self.get_total(component)
            plt.plot(data_total.columns.values, data_total.values.transpose())

        plt.show()

    @classmethod
    def edit_nodes_v2(cls, data, node_edit):
        if "node" not in data.index.dtypes and "location" not in data.index.dtypes:
            return data
        #check if data of specific node, specified by string, should be extracted
        if isinstance(node_edit,str):
            if data.index.nlevels == 2:
                data = data.loc[(slice(None), node_edit), :]
                return data
            elif data.index.nlevels == 3:
                data = data.loc[(slice(None), slice(None), node_edit), :]
                return data
            elif data.index.nlevels == 4:
                return
        #check if data varying only in node value should be summed
        elif node_edit == True:
            level_names = {}
            for level in data.index.levels:
                if level.name not in ["node", "location"]:
                    level_names[level.name] = level.values
            if len(level_names) == 2:
                for level_name_1 in level_names[0]:
                    for level_name_2 in level_names[1]:
                        data = data
    @classmethod
    def extract_carrier(cls, data, carrier):
        if "carrier" not in data.index.dtypes:
            if "technology" not in data.index.dtypes:
                return data
        if data.index.nlevels == 2:
            data = data.loc[(carrier, slice(None)), :]
            return data
        elif data.index.nlevels == 3:
            data = data.loc[(slice(None), carrier, slice(None)), :]
            return data
        elif data.index.nlevels == 4:
            return

    @classmethod
    def edit_nodes(cls, data, node_edit):
        """
        Either returns data of a single node or sums data of identical indices at different nodes
        :param data: data frame of component
        :param node_edit: Either True (sum over nodes) or a string specifying a specific node
        """
        index_list = []
        #check if data of a specific node should be returned
        if isinstance(node_edit, str):
            for ind, name in enumerate(data.index.names):
                if name in ["node", "location"]:
                    index_list.extend([index for index in data.index if index[ind] == node_edit])
            return data.loc[data.index.isin(index_list)]

        #return data as it doesn't have multiple levels (nothing can be summed up)
        if data.index.nlevels == 1:
            return data
        #check if data contains nodes
        if "node" not in data.index.dtypes and "location" not in data.index.dtypes:
            return data
        multi_index_list = []
        data_nodes_summed = pd.DataFrame()
        #check if data contains technology and carrier index levels as it is the case for: input_flow, output_flow, ...
        if "technology" in data.index.dtypes and "carrier" in data.index.dtypes:
            #sort data such that identical technologies and carries lie next to each other
            data = data.sort_index(axis=0,level=["technology","carrier"])
            #iterate over rows of data to find entries with same technologies and carriers at multiple nodes
            for pos, index in enumerate(data.index):
                #ensure that data isn't accessed out of bound and assess last row of data
                if pos == data.index.shape[0] - 1:
                    index_list.append(index)
                    test_data = data.loc[data.index.isin(index_list)].sum(axis=0)
                    data_nodes_summed = pd.concat([data_nodes_summed,test_data],axis=1)
                    multi_index_list.append((index[0],index[1]))
                #check if technology and carrier at current row match the ones from the next row
                elif index[0] == data.index[pos+1][0] and index[1] == data.index[pos+1][1]:
                    #store index to sum corresponding value later on
                    index_list.append(index)
                #sum the values of identical technologies and carriers of different nodes
                else:
                    index_list.append(index)
                    #sum values
                    test_data = data.loc[data.index.isin(index_list)].sum(axis=0)
                    #add sum to data frame with summed values of other technologies and carriers
                    data_nodes_summed = pd.concat([data_nodes_summed,test_data],axis=1)
                    #empty index_list such that indices of next technology/carrier combination can be stored
                    index_list = []
                    #store index of current technology/carrier combination such that data_nodes_summed can be indexed later on
                    multi_index_list.append((index[0],index[1]))
            #transpose DataFrame to overcome concat limitation
            data_nodes_summed = data_nodes_summed.transpose()
            #create MultiIndex of the gathered technology/carrier combinations
            multi_index = pd.MultiIndex.from_tuples(multi_index_list, names=["technology","carrier"])
            #set multi index
            data_nodes_summed = data_nodes_summed.set_index(multi_index)
            return data_nodes_summed

        #check if data contains technology and capacity_type index levels as it is the case for: capacity, ...
        elif "technology" in data.index.dtypes and "capacity_type" in data.index.dtypes:
            #sort data such that identical technologies and capacity_types lie next to each other
            data = data.sort_index(axis=0,level=["technology","capacity_type"])
            #iterate over rows of data to find entries with same technologies and carriers at multiple locations
            for pos, index in enumerate(data.index):
                #ensure that data isn't accessed out of bound and assess last row of data
                if pos == data.index.shape[0] - 1:
                    index_list.append(index)
                    test_data = data.loc[data.index.isin(index_list)].sum(axis=0)
                    data_nodes_summed = pd.concat([data_nodes_summed,test_data],axis=1)
                    multi_index_list.append((index[0],index[1]))
                #check if technology and carrier at current row match the ones from the next row
                elif index[0] == data.index[pos+1][0] and index[1] == data.index[pos+1][1]:
                    #store index to sum corresponding value later on
                    index_list.append(index)
                #sum the values of identical technologies and carriers of different nodes
                else:
                    index_list.append(index)
                    #sum values
                    test_data = data.loc[data.index.isin(index_list)].sum(axis=0)
                    #add sum to data frame with summed values of other technologies and carriers
                    data_nodes_summed = pd.concat([data_nodes_summed,test_data],axis=1)
                    #empty index_list such that indices of next technology/carrier combination can be stored
                    index_list = []
                    #store index of current technology/carrier combination such that data_nodes_summed can be indexed later on
                    multi_index_list.append((index[0],index[1]))
            #transpose DataFrame to overcome concat limitation
            data_nodes_summed = data_nodes_summed.transpose()
            #create MultiIndex of the gathered technology/carrier combinations
            multi_index = pd.MultiIndex.from_tuples(multi_index_list, names=["technology","capacity_type"])
            #set multi index
            data_nodes_summed = data_nodes_summed.set_index(multi_index)
            return data_nodes_summed
        else:
            warnings.warn("further implementation needed to sum over nodes of this variable")
            return data
    @classmethod
    def sum_over_technologies(cls, data):
        #return data as it doesn't have multiple levels (nothing can be summed up)
        if data.index.nlevels == 1:
            return data
        #return data as technology is not contained in the index levels
        if "technology" not in data.index.dtypes:
            return data
        index_list = []
        multi_index_list = []
        data_technologies_summed = pd.DataFrame()
        #check if data contains technology and carrier index levels as it is the case for: input_flow, output_flow, ...
        if "technology" in data.index.dtypes and "carrier" in data.index.dtypes:

            if "node" in data.index.dtypes:
                #sort data such that identical carries and nodes lie next to each other
                data = data.sort_index(axis=0, level=["carrier","node"])
                #iterate over rows of data to find technologies with identical carrier
                for pos, index in enumerate(data.index):
                    #ensure that data isn't accessed out of bound and assess last row of data
                    if pos == data.index.shape[0] - 1:
                        index_list.append(index)
                        test_data = data.loc[data.index.isin(index_list)].sum(axis=0)
                        data_technologies_summed = pd.concat([data_technologies_summed,test_data],axis=1)
                        multi_index_list.append((index[1],index[2]))
                    #check if technology and carrier at current row match the ones from the next row
                    elif index[1] == data.index[pos+1][1] and index[2] == data.index[pos+1][2]:
                        #store index to sum corresponding value later on
                        index_list.append(index)
                    #sum the values of identical technologies and carriers of different nodes
                    else:
                        index_list.append(index)
                        #sum values
                        test_data = data.loc[data.index.isin(index_list)].sum(axis=0)
                        #add sum to data frame with summed values of other technologies and carriers
                        data_technologies_summed = pd.concat([data_technologies_summed,test_data],axis=1)
                        #empty index_list such that indices of next technology/carrier combination can be stored
                        index_list = []
                        #store index of current technology/carrier combination such that data_technologies_summed can be indexed later on
                        multi_index_list.append((index[1],index[2]))
                #transpose DataFrame to overcome concat limitation
                data_technologies_summed = data_technologies_summed.transpose()
                #create MultiIndex of the gathered technology/carrier combinations
                multi_index = pd.MultiIndex.from_tuples(multi_index_list, names=["carrier","node"])
                #set multi index
                data_technologies_summed = data_technologies_summed.set_index(multi_index)
                return data_technologies_summed
            #data doesn't have a node index
            else:
                # sort data such that identical carries lie next to each other
                data = data.sort_index(axis=0, level=["carrier"])
                # iterate over rows of data to find technologies with identical carrier
                for pos, index in enumerate(data.index):
                    # ensure that data isn't accessed out of bound and assess last row of data
                    if pos == data.index.shape[0] - 1:
                        index_list.append(index)
                        test_data = data.loc[data.index.isin(index_list)].sum(axis=0)
                        data_technologies_summed = pd.concat([data_technologies_summed, test_data], axis=1)
                        multi_index_list.append(index[1])
                    # check if technology and carrier at current row match the ones from the next row
                    elif index[1] == data.index[pos + 1][1]:
                        # store index to sum corresponding value later on
                        index_list.append(index)
                    # sum the values of identical technologies and carriers of different nodes
                    else:
                        index_list.append(index)
                        # sum values
                        test_data = data.loc[data.index.isin(index_list)].sum(axis=0)
                        # add sum to data frame with summed values of other technologies and carriers
                        data_technologies_summed = pd.concat([data_technologies_summed, test_data], axis=1)
                        # empty index_list such that indices of next technology/carrier combination can be stored
                        index_list = []
                        # store index of current technology/carrier combination such that data_technologies_summed can be indexed later on
                        multi_index_list.append(index[1])
                # transpose DataFrame to overcome concat limitation
                data_technologies_summed = data_technologies_summed.transpose()
                data_technologies_summed = data_technologies_summed.set_index([multi_index_list])
                return data_technologies_summed

        else:
            warnings.warn("Technologies using same carrier cannot be summed as variable doesn't have a technology AND a carrier index")
            return data

    def extract_technology(self, data, type):
        #check if data contains technologies
        if "technology" not in data.index.dtypes:
            return data
        index_list = []
        #check if data contains technology and capacity_type index levels as it is the case for: capacity, ...
        if "technology" in data.index.dtypes and "capacity_type" in data.index.dtypes:
            if "location" in data.index.dtypes:
                #iterate over rows of data to find technologies with identical carrier
                for pos, index in enumerate(data.index):
                    if type == "conversion":
                        if index[0] in self.results["system"]["set_conversion_technologies"]:
                            index_list.append(index)
                    elif type == "transport":
                        if index[0] in self.results["system"]["set_transport_technologies"]:
                            index_list.append(index)
                    elif "storage" in type:
                        if index[0] in self.results["system"]["set_storage_technologies"]:
                            if "power" in type and index[1] == "power":
                                index_list.append(index)
                            elif "energy" in type and index[1] == "energy":
                                index_list.append(index)
                            elif "power" not in type and "energy" not in type:
                                index_list.append(index)
                    else:
                        warnings.warn("Technology type doesn't exist!")
        return data.loc[data.index.isin(index_list)]

    def extract_reference_carrier(self, data, type):
        """
        Extracts technologies of reference carrier type
        :param data: Data Frame containing set of technologies with different reference carriers
        :param type: String specifying reference carrier whose technologies should be extracted from data
        :return: Data Frame containing technologies of reference carrier only
        """
        _output_carriers = self.get_df("output_flow").index.droplevel(
            level=["node", "time_operation"]).unique().to_frame().set_index("technology")
        index_list = []
        for tech, carrier in enumerate(_output_carriers["carrier"]):
            if carrier == type:
                index_list.extend([index for index in data.index if index[0] == _output_carriers.index[tech]])

        return data.loc[data.index.isin(index_list)]

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