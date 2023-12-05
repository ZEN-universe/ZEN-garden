"""
:Title:        ZEN-GARDEN
:Created:      October-2021
:Authors:      Alissa Ganter (aganter@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Class is defining to read in the results of an Optimization problem.
"""

import importlib
import json
import logging
import os
import zlib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import warnings
# from tables import NaturalNameWarning
from tqdm import tqdm
from zen_garden import utils
from zen_garden.model.objects.time_steps import TimeStepsDicts
import h5py
from typing import Any

# Warnings
# warnings.filterwarnings('ignore', category=NaturalNameWarning)

# SETUP LOGGER
utils.setup_logger()


class TimeStepDictFromFile:
    def __init__(self, time_dict: dict[Any, Any], scenario: str):
        self.time_dict = time_dict
        self.scenario = "default" if scenario is None else scenario
        self.sequence_time_steps_cache: dict[str, npt.NDArray[np.float_]] = {}
        self.year2operation_cache: dict[str, pd.DataFrame] = {}
        self.year2storage_cache: dict[str, pd.DataFrame] = {}

    def get_time_steps_year2operation(self, year: int) -> pd.DataFrame:
        hash_key = str(year)
        if hash_key not in self.year2operation_cache:
            self.year2operation_cache[hash_key] = Results._to_df(
                self.time_dict["time_steps_year2operation"][str(year)]
            )
        return self.year2operation_cache[hash_key]

    def get_time_steps_year2operation_old(self, tech_proxy:str,year: int) -> pd.DataFrame:
        hash_key = tech_proxy + str(year)
        if hash_key not in self.year2operation_cache:
            self.year2operation_cache[hash_key] = Results._to_df(
                self.time_dict["time_steps_year2operation"][tech_proxy][str(year)]
            )
        return self.year2operation_cache[hash_key]

    def get_time_steps_year2storage(self, year: int) -> pd.DataFrame:
        hash_key = str(year)
        if hash_key not in self.year2storage_cache:
            self.year2storage_cache[hash_key] = Results._to_df(
                self.time_dict["time_steps_year2storage"][str(year)]
            )
        return self.year2storage_cache[hash_key]

    def get_sequence_time_steps(self,time_step_type = "operation") -> npt.NDArray[np.float_]:
        """
        Get sequence ot time steps of element

        :param time_step_type: type of time step (operation, storage or yearly)
        :return sequence_time_steps: list of time steps corresponding to base time step
        """
        if time_step_type not in self.sequence_time_steps_cache:
            if time_step_type not in self.time_dict:
                sequence_storage = [v for k,v in self.time_dict["operation"].items() if "storage_level" in k]
                if time_step_type == "storage" and len(sequence_storage) > 0:
                    self.sequence_time_steps_cache[time_step_type] = sequence_storage[0]["values"][:]
                else:
                    raise KeyError(f"Time step type {time_step_type} is incorrect")
            else:
                self.sequence_time_steps_cache[time_step_type] = self.time_dict[time_step_type]["values"][:]
        return self.sequence_time_steps_cache[time_step_type]

class Results(object):
    """
    This class reads in the results after the pipeline has run
    """

    def __init__(self, path, scenarios=None, load_opt=False):
        """
        Initializes the Results class with a given path

        :param path: Path to the output of the optimization problem
        :param scenarios: A None, str or tuple of scenarios to load, defaults to all scenarios
        :param load_opt: Optionally load the opt dictionary as well
        """

        # get the abs path
        self.path = os.path.abspath(path)
        self.name = Path(path).name
        # check if the path exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"No such file or directory: {self.path}")

        # load the onetime stuff
        self.results = {}
        self.results["scenarios"] = self.load_scenarios(self.path)
        self.component_names = {
            "pars": {},
            "vars": {},
            "sets": {},
            "duals": {},
        }

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
        elif [folder.name for folder in os.scandir(self.path) if folder.is_dir() and "scenario_" in folder.name]:
            self.has_scenarios = True
            self.scenarios = [f"scenario_{scenario}" for scenario in self.results["scenarios"].keys()]
        # there are no scenarios
        else:
            self.has_scenarios = False
            self.scenarios = [None]

        # cycle through the dirs
        for scenario in self.scenarios:
            # init dict
            self.results[scenario] = {}
            # get the base scenario
            base_scenario = ""
            scenario_subfolder = None
            # path to access scenario-dependent files
            file_folder = Path("")
            if self.has_scenarios:
                # name without scenario_ prefix
                name_short = scenario[9:]
                base_scenario = self.results['scenarios'][name_short].get('base_scenario', '')
                scenario_subfolder = self.results['scenarios'][name_short].get('sub_folder', '')
                if scenario_subfolder == '':
                    # no scenario subfolder -> no need to switch directories
                    base_scenario = ""
                    scenario_subfolder = scenario
                    file_folder = scenario_subfolder
                else:
                    # we need to go one level deeper
                    base_scenario = f"scenario_{base_scenario}"
                    scenario_subfolder = f"scenario_{scenario_subfolder}"
                    file_folder = os.path.join(base_scenario, scenario_subfolder)

            # load scenario-dependent files
            self.results[scenario]["system"] = self.load_system(os.path.join(self.path, file_folder))
            self.results[scenario]["analysis"] = self.load_analysis(os.path.join(self.path, file_folder))
            self.results[scenario]["solver"] = self.load_solver(os.path.join(self.path, file_folder))
            # myopic foresight
            if self.results[scenario]["system"]["use_rolling_horizon"]:
                self.results[scenario]["has_MF"] = True
                self.results[scenario]["mf"] = [f"MF_{step_horizon}" for step_horizon in self.get_years(scenario)]
            else:
                self.results[scenario]["has_MF"] = False
                self.results[scenario]["mf"] = [None]

            # load the corresponding timestep dict
            sub_path = os.path.join(self.path, base_scenario)
            time_dict = self.load_sequence_time_steps(sub_path, scenario_subfolder, lazy=True)
            # if updated time steps
            if "storage" in time_dict:
                self.new_time_steps = True
            else:
                self.new_time_steps = False
                logging.warning("Old time step dict will be deprecated. Please rerun results.")
            timesteps_are_precalculated = "time_steps_year2operation" in time_dict
            if timesteps_are_precalculated:
                self.results[scenario]["sequence_time_steps_dicts"] = TimeStepDictFromFile(time_dict, scenario)
            else:
                time_dict = self.load_sequence_time_steps(sub_path, scenario_subfolder, lazy=False)
                if not self.new_time_steps:
                    self.results[scenario]["sequence_time_steps_dicts"] = TimeStepsDicts(time_dict)
                    # load the operation2year and year2operation time steps
                    self.results[scenario]["sequence_time_steps_dicts"].set_time_steps_operation2year_both_dir()
                    self.results[scenario]["sequence_time_steps_dicts"].set_time_steps_storage2year_both_dir()
                else:
                    raise NotImplementedError("New time step structure without precalculated sequences is not yet implemented")

            # self.results[scenario]["dict_sequence_time_steps"] = time_dict

            for mf in self.results[scenario]["mf"]:
                # init dict
                self.results[scenario][mf] = {}
                # deal with MF
                if self.results[scenario]["has_MF"] and self.results[scenario]["system"]["optimized_years"] > 1:
                    if self.has_scenarios:
                        scenfolder = Path(scenario_subfolder)
                        subfolder = os.path.join(scenfolder, Path(mf))
                    else:
                        subfolder = Path(mf)
                    # Add together
                    current_path = os.path.join(sub_path, subfolder)
                else:
                    current_path = Path(sub_path)
                    if self.has_scenarios:
                        scenfolder = Path(scenario_subfolder)
                        current_path = os.path.join(current_path,scenfolder)

                # create dict containing sets
                self.results[scenario][mf]["sets"] = {}
                sets = self.load_sets(current_path, lazy=True)
                self._lazydicts.append(sets)
                if not self.component_names["sets"]:
                    self.component_names["sets"] = list(sets.keys())
                self.results[scenario][mf]["sets"].update(sets)

                # create dict containing params and vars
                self.results[scenario][mf]["pars_and_vars"] = {}
                pars = self.load_params(current_path, lazy=True)
                self._lazydicts.append(pars)
                if not self.component_names["pars"]:
                    self.component_names["pars"] = list(pars.keys())
                self.results[scenario][mf]["pars_and_vars"].update(pars)
                vars = self.load_vars(current_path, lazy=True)
                self._lazydicts.append(vars)
                if not self.component_names["vars"]:
                    self.component_names["vars"] = list(vars.keys())
                self.results[scenario][mf]["pars_and_vars"].update(vars)

                # load duals
                if self.results[scenario]["solver"]["add_duals"]:
                    duals = self.load_duals(current_path, lazy=True)
                    self._lazydicts.append(duals)
                    if not self.component_names["duals"]:
                        self.component_names["duals"] = list(duals.keys())
                    self.results[scenario][mf]["duals"] = duals
                # the opt we only load when requested
                if load_opt:
                    self.results[scenario][mf]["optdict"] = self.load_opt(current_path)

        # load the time step duration, these are normal dataframe calls (dicts in case of scenarios)
        self.time_step_operational_duration = self.load_time_step_operation_duration()
        self.time_step_storage_duration = self.load_time_step_storage_duration()

    def close(self):
        """
        Close all open handles
        """

        for lazydict in self._lazydicts:
            lazydict.close()

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
            if not lazy:
                content = utils.HDFPandasSerializer(file_name=f"{name}.h5", lazy=lazy)
            else:
                content = h5py.File(f"{name}.h5")
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

        # nothing to do
        if isinstance(dict_raw, utils.HDFPandasSerializer):
            return dict_raw

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

        # nothing to do
        if isinstance(string, (pd.DataFrame, pd.Series)):
            return string

        if isinstance(string, h5py.Group):
            return pd.read_hdf(string.file.filename, string.name)

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
        """Loads the analysis dict from a given path

        :param path: Directory to load the dictionary from
        :return: The analysis dictionary
        """

        # get the dict
        raw_dict = cls._read_file(os.path.join(path, "analysis"))

        return raw_dict

    @classmethod
    def load_solver(cls, path):
        """Loads the solver dict from a given path

        :param path: Directory to load the dictionary from
        :return: The analysis dictionary
        """

        # get the dict
        raw_dict = cls._read_file(os.path.join(path, "solver"))

        return raw_dict

    @classmethod
    def load_scenarios(cls, path):
        """Loads the scenarios dict from a given path

        :param path: Directory to load the dictionary from
        :return: The analysis dictionary
        """

        # get the dict
        raw_dict = cls._read_file(os.path.join(path, "scenarios"))

        return raw_dict

    @classmethod
    def load_opt(cls, path):
        """Loads the opt dict from a given path

        :param path: Directory to load the dictionary from
        :return: The analysis dictionary
        """

        # get the dict
        raw_dict = cls._read_file(os.path.join(path, "opt_dict"))

        return raw_dict

    @classmethod
    def load_sequence_time_steps(cls, path, scenario=None, lazy=False):
        """Loads the dict_sequence_time_steps from a given path

        :param path: Path to load the dict from
        :param scenario: Name of the scenario to load
        :return: dict_sequence_time_steps
        """
        # get the file name
        if scenario:
            path = os.path.join(path, scenario)
        fname = os.path.join(path, "dict_all_sequence_time_steps")
        if scenario is not None:
            fname += f"_{scenario}"

        # get the dict
        dict_sequence_time_steps = cls._read_file(fname, lazy=lazy)

        # tranform all lists to arrays
        return cls.expand_dict(dict_sequence_time_steps)

    @classmethod
    def expand_dict(cls, dictionary):
        """Creates a copy of the dictionary where all lists are recursively transformed to numpy arrays

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

    @classmethod
    def compare_configs(cls, results: list, scenarios=None):
        """Compares the configs of two or more results

        :param results: list of results
        :param scenarios: None, str or tuple of scenarios
        :return: a dictionary with diverging results
        """
        results, scenarios = cls.check_combine_results(results, scenarios)
        if results is None:
            return
        result_names = [res.name for res in results]
        logging.info(f"Comparing the configs of {result_names}")
        diff_analysis = cls.get_config_diff(results, "analysis", scenarios)
        diff_system = cls.get_config_diff(results, "system", scenarios)
        diff_solver = cls.get_config_diff(results, "solver", scenarios)

        diff_dict = {}
        if diff_analysis:
            diff_dict["analysis"] = diff_analysis
        if diff_system:
            diff_dict["system"] = diff_system
        if diff_solver:
            diff_dict["solver"] = diff_solver
        return diff_dict

    @classmethod
    def compare_model_parameters(cls, results: list, compare_total=True, scenarios=None):
        """Compares the input data of two or more results

        :param results: list of results
        :param compare_total: if True, compare total value, not full time series
        :param scenarios: None, str or tuple of scenarios
        :return: a dictionary with diverging results
        """
        results, scenarios = cls.check_combine_results(results, scenarios)
        if results is None:
            return
        result_names = [res.name for res in results]
        logging.info(f"Comparing the model parameters of {result_names}")
        diff_pars = cls.get_component_diff(results,"pars")
        diff_dict = {}
        # initialize progress bar
        pbar = tqdm(total=len(diff_pars))
        for par in diff_pars:
            # update progress bar
            pbar.update(1)
            pbar.set_description(f"Compare parameter {par}")
            # check par
            comparison_df = cls.compare_component_values(results, par, compare_total, scenarios=scenarios)
            if not comparison_df.empty:
                logging.info(f"Parameter {par} has different values")
                diff_dict[par] = comparison_df
        pbar.close()
        return diff_dict

    @classmethod
    def compare_model_variables(cls, results: list, compare_total=True, scenarios=None):
        """Compares the input data of two or more results

        :param results: list of results
        :param compare_total: if True, compare total value, not full time series
        :param scenarios: None, str or tuple of scenarios
        :return: a dictionary with diverging results
        """
        results,scenarios = cls.check_combine_results(results,scenarios)
        if results is None:
            return
        result_names = [res.name for res in results]
        logging.info(f"Comparing the model variables of {result_names}")
        diff_vars = cls.get_component_diff(results,"vars")
        diff_dict = {}
        # initialize progress bar
        pbar = tqdm(total = len(diff_vars))
        for var in diff_vars:
            # update progress bar
            pbar.update(1)
            pbar.set_description(f"Compare variable {var}")
            # check var
            comparison_df = cls.compare_component_values(results, var, compare_total, scenarios=scenarios)
            if not comparison_df.empty:
                logging.info(f"Variable {var} has different values")
                diff_dict[var] = comparison_df
        pbar.close()
        return diff_dict

    @classmethod
    def compare_component_values(cls, results, component, compare_total, scenarios, rtol=1e-3):
        """Compares component values of two results

        :param results: list with results
        :param component: component name
        :param compare_total: if True, compare total value, not full time series
        :param scenarios: None, str or tuple of scenarios
        :param rtol: relative tolerance of equal values
        :return: dictionary with diverging component values
        """
        result_names = [res.name for res in results]
        if compare_total:
            val_0 = results[0].get_total(component,scenario=scenarios[0])
            val_1 = results[1].get_total(component,scenario=scenarios[1])
        else:
            val_0 = results[0].get_full_ts(component,scenario=scenarios[0])
            val_1 = results[1].get_full_ts(component,scenario=scenarios[1])
        mismatched_index = False
        if isinstance(val_0, pd.DataFrame):
            val_0 = val_0.sort_index(axis=0).sort_index(axis=1)
            val_1 = val_1.sort_index(axis=0).sort_index(axis=1)
            if not val_0.index.equals(val_1.index) or not val_0.columns.equals(val_1.columns):
                mismatched_index = True
        else:
            val_0 = val_0.sort_index()
            val_1 = val_1.sort_index()
            if not val_0.index.equals(val_1.index):
                mismatched_index = True
        if mismatched_index:
            logging.info(f"Component {component} does not have matching index or columns")
            comparison_df = pd.concat([val_0,val_1],keys=result_names,axis=1)
            comparison_df = comparison_df.sort_index(axis=1, level=1)
            return comparison_df
        is_close = np.isclose(val_0,val_1,rtol=rtol,equal_nan=True)
        if isinstance(val_0,pd.DataFrame):
            diff_val_0 = val_0[(~is_close).any(axis=1)]
            diff_val_1 = val_1[(~is_close).any(axis=1)]
        else:
            diff_val_0 = val_0[(~is_close)]
            diff_val_1 = val_1[(~is_close)]
        comparison_df = pd.concat([diff_val_0,diff_val_1],keys=result_names,axis=1)
        comparison_df = comparison_df.sort_index(axis=1,level=1)
        return comparison_df

    @classmethod
    def check_combine_results(cls, results: list, scenarios=None):
        """Checks if results are a list of 2 results with the matching scenarios

        :param results: list of results
        :param scenarios: None, str or tuple of scenarios
        :return: dictionary of results
        """
        if len(results) != 2:
            logging.warning("You must select exactly two results to compare. Skip.")
            return None, None
        if type(results) != list:
            logging.warning("You must pass the results as list. Skip")
            return None,None
        for el in results:
            if type(el) != cls:
                logging.warning(f"You must pass a list of ZEN-garden results, not type {type(el)}. Skip")
                return None,None
        scenarios = cls.check_scenario_results(results, scenarios)
        return results, scenarios

    @classmethod
    def check_scenario_results(cls,results:list,scenarios=None):
        """Checks if results have scenarios and if yes, if the provided scenarios match

        :param results: list of results
        :param scenarios: None, str or tuple of scenarios
        :return: scenarios
        """
        # neither result has scenarios
        if not results[0].has_scenarios and not results[1].has_scenarios:
            scenarios = (None,None)
            return scenarios
        # if scenarios is None, choose base scenario for both
        elif scenarios is None:
            scenarios = (results[0].scenarios[0], results[1].scenarios[0])
            logging.info(f"At least one result has scenarios but no scenarios are provided. Scenarios {scenarios} are selected")
        # if one scenario string is provided
        elif type(scenarios) == str:
            if scenarios not in results[0].scenarios:
                scenario_0 = results[0].scenarios[0]
                if results[0].has_scenarios:
                    logging.info(f"Scenario {scenarios} not in scenarios of {results[0].name} ({results[0].scenarios}). Scenario {scenario_0} is selected")
            else:
                scenario_0 = scenarios
            if scenarios not in results[1].scenarios:
                scenario_1 = results[1].scenarios[0]
                if results[1].has_scenarios:
                    logging.info(
                        f"Scenario {scenarios} not in scenarios of {results[1].name} ({results[1].scenarios}). Scenario {scenario_1} is selected")
            else:
                scenario_1 = scenarios
            scenarios = (scenario_0,scenario_1)
        # if scenarios is tuple
        elif type(scenarios) == tuple:
            if scenarios[0] not in results[0].scenarios:
                scenario_0 = results[0].scenarios[0]
                if results[0].has_scenarios:
                    logging.info(f"Scenario {scenarios[0]} not in scenarios of {results[0].name} ({results[0].scenarios}). Scenario {scenario_0} is selected")
            else:
                scenario_0 = scenarios[0]
            if scenarios[1] not in results[1].scenarios:
                scenario_1 = results[1].scenarios[0]
                if results[1].has_scenarios:
                    logging.info(
                        f"Scenario {scenarios[1]} not in scenarios of {results[1].name} ({results[1].scenarios}). Scenario {scenario_1} is selected")
            else:
                scenario_1 = scenarios[1]
            scenarios = (scenario_0,scenario_1)
        else:
            raise TypeError(f"Scenarios must be of type <str> or <tuple> not {type(scenarios)}")
        return scenarios

    @classmethod
    def get_config_diff(cls, results, config_type, scenarios):
        """returns a dict with the differences in config_type values

        :param results: dictionary with results
        :param config_type: name of result config_type
        :return: Dictionary with differences in config_type values
        """
        result_names = [res.name for res in results]
        diff_dict = cls.compare_dicts(
            results[0]._get_config_dict(config_type,scenarios[0]),
            results[1]._get_config_dict(config_type,scenarios[0]), result_names)
        return diff_dict

    @classmethod
    def compare_dicts(cls,dict1,dict2,result_names):
        """

        :param dict1: first config dict
        :param dict2: second config dict
        :param result_names: names of results
        :return: diff dict
        """
        diff_dict = {}
        for key in dict1.keys() | dict2.keys():
            if isinstance(dict1.get(key), dict) and isinstance(dict2.get(key), dict):
                nested_diff = cls.compare_dicts(dict1.get(key, {}), dict2.get(key, {}),result_names)
                if nested_diff:
                    diff_dict[key] = nested_diff
            elif dict1.get(key) != dict2.get(key):
                if isinstance(dict1.get(key),list) and isinstance(dict2.get(key),list):
                    if sorted(dict1.get(key)) != sorted(dict2.get(key)):
                        diff_dict[key] = {result_names[0]:sorted(dict1.get(key)), result_names[1]:sorted(dict2.get(key))}
                else:
                    diff_dict[key] = {result_names[0]:dict1.get(key), result_names[1]:dict2.get(key)}
        return diff_dict if diff_dict else None

    @staticmethod
    def get_component_diff(results,component_type):
        """returns a list with the differences in component names

        :param results: dictionary with results
        :return: list with the common params
        """
        result_names = [res.name for res in results]
        list_component = [res.component_names[component_type] for res in results]
        only_in_0 = set(list_component[0]).difference(list_component[1])
        only_in_1 = set(list_component[1]).difference(list_component[0])
        common_component = sorted(list(set(list_component[0]).intersection(list_component[1])))
        if only_in_1 and only_in_0:
            logging.info(f"Components {only_in_1} are missing from {result_names[0]} and "
                         f"parameters {only_in_0} are missing from {result_names[1]}")
        elif only_in_1:
            logging.info(f"Components {only_in_1} are missing from {result_names[0]}")
        elif only_in_0:
            logging.info(f"Components {only_in_0} are missing from {result_names[1]}")
        return common_component

    def get_df(self, name, scenario=None, to_csv=None, csv_kwargs=None,is_dual=False, is_set=False):
        """Extracts the dataframe from the results

        :param name: The name of the dataframe to extract
        :param scenario: If multiple scenarios are in the results, only consider this one
        :param to_csv: Save the dataframes to a csv file
        :param csv_kwargs: additional keyword arguments forwarded to the to_csv method of pandas
        :param is_dual: if dual variable dict is selected
        :param is_set: if sets are selected
        :return: The dataframe that should have been extracted. If multiple scenarios are present a dictionary
                 with scenarios as keys and dataframes as value is returned
        """

        # select the scenarios
        if scenario is not None:
            scenarios = [scenario]
        else:
            scenarios = self.scenarios

        # loop
        data = {}
        for scenario in scenarios:
            # we get the timestep dict
            sequence_time_steps_dicts = self.results[scenario]["sequence_time_steps_dicts"]

            if not self.results[scenario]["has_MF"]:
                if not is_set:
                    # we set the dataframe of the variable into the data dict
                    if not is_dual:
                        data[scenario] = self._to_df(self.results[scenario][None]["pars_and_vars"][name]["dataframe"])
                    else:
                        data[scenario] = self._to_df(self.results[scenario][None]["duals"][name]["dataframe"])
                else:
                    data[scenario] = self._to_df(self.results[scenario][None]["sets"][name]["dataframe"])

            else:
                # init the scenario
                mf_data = {}
                is_multiindex = False
                # cycle through all MFs
                for year, mf in enumerate(self.results[scenario]["mf"]):
                    if not is_set:
                        if not is_dual:
                            var = self._to_df(self.results[scenario][mf]["pars_and_vars"][name]["dataframe"])
                        else:
                            var = self._to_df(self.results[scenario][mf]["duals"][name]["dataframe"])
                    else:
                        var = self._to_df(self.results[scenario][mf]["sets"][name]["dataframe"])

                    # no multiindex
                    if var.index.nlevels == 1:
                        ts_type = self._get_ts_type(var.T, name, scenario=scenario, force_output=True)
                        # if yearly variable
                        if ts_type == "yearly":
                            mf_data[year] = var.loc[year].squeeze()
                            yearly_component = True
                            time_header = var.index.name
                        elif ts_type is not None:
                            # get the timesteps
                            time_steps_year = self._get_time_steps_of_year(sequence_time_steps_dicts,ts_type,year)
                            mf_data[year] = var.loc[time_steps_year].squeeze()
                            yearly_component = False
                            time_header = var.index.name
                        else:
                            data[scenario] = var
                            break
                    # multiindex
                    else:
                        is_multiindex = True
                        # unstack the year
                        var_series = var["value"].unstack()
                        # get type of time steps
                        ts_type = self._get_ts_type(var_series, name, scenario=scenario, force_output=True)
                        if ts_type == "yearly":
                            # get the data
                            tmp_data = var_series[year]
                            # rename
                            tmp_data.name = var_series.columns.name
                            # set
                            mf_data[year] = tmp_data
                            yearly_component = True
                            time_header = var_series.columns.name
                        # operational ts (we drop value in columns)
                        elif ts_type is not None:
                            # get the timesteps
                            time_steps_year = self._get_time_steps_of_year(sequence_time_steps_dicts,ts_type,year)
                            # get the data
                            tmp_data = var_series[[tstep for tstep in time_steps_year]]
                            # rename
                            tmp_data.name = var_series.columns.name
                            # set
                            mf_data[year] = tmp_data
                            yearly_component = False
                            time_header = var_series.columns.name
                        # else not a time index
                        else:
                            data[scenario] = var_series.stack()
                            break
                # This is a for-else, it is triggered if we did not break the loop
                else:
                    # deal with the years
                    if yearly_component:
                        # concat
                        if is_multiindex:
                            new_header = [time_header]+tmp_data.index.names
                            new_order = tmp_data.index.names + [time_header]
                            df = pd.concat(mf_data, axis=0, keys=mf_data.keys(),names=new_header).reorder_levels(new_order)
                        else:
                            df = pd.Series(mf_data,index=mf_data.keys())
                            df.index.name = time_header

                    else:
                        if isinstance(mf_data[list(mf_data.keys())[0]],pd.Series):
                            df = pd.concat(mf_data)
                            df.index = df.index.droplevel(0)
                            df = df.sort_index()
                        elif isinstance(mf_data[list(mf_data.keys())[0]],pd.DataFrame):
                            df = pd.concat(mf_data, axis=1)
                            df.columns = df.columns.droplevel(0)
                            df = df.sort_index(axis=1).stack()
                        else:
                            df = pd.Series(mf_data)

                    data[scenario] = df

        # transform all dataframes to pd.Series with the element_name as name
        for k, v in data.items():
            if not isinstance(v, pd.Series):
                # to series
                series = pd.Series(data=v["value"], index=v.index)
                series.name = name
                # set
                data[k] = series
            # we just make sure the name is right
            else:
                v.name = name
                data[k] = v

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
                data[scenario].to_csv(f"{fname}.csv", **csv_kwargs)
            return data[scenario]

        # return the dict or series
        else:
            # save if necessary
            if to_csv is not None:
                for scenario in scenarios:
                    data[scenario].to_csv(f"{fname}_{scenario}.csv", **csv_kwargs)
            return data

    def _single_operational_df(self):
        """ extracts a single operational df"""

    def load_time_step_operation_duration(self):
        """
        Loads duration of operational time steps
        """
        d = self.get_df("time_steps_operation_duration")
        if not self.new_time_steps:
            if self.has_scenarios:
                # TODO make time_step_operation_duration scenario-dependent
                d = d[self.scenarios[0]].unstack().iloc[0]
            else:
                d = d.unstack().iloc[0]
        return d

    def load_time_step_storage_duration(self):
        """
        Loads duration of operational time steps
        """
        if not self.new_time_steps:
            d = self.get_df("time_steps_storage_level_duration")
            if self.has_scenarios:
                # TODO make time_step_storage_duration scenario-dependent
                d = d[self.scenarios[0]].unstack().iloc[0]
            else:
                d = d.unstack().iloc[0]
        else:
            d = self.get_df("time_steps_storage_duration")
        return d

    def get_full_ts(
            self,
            component,
            element_name=None,
            year=None,
            scenario=None,
            is_dual=False,
            discount_to_first_step=True,
            node=None,
            start_time_step=0,
            end_time_step=None):
        """Calculates the full timeseries for a given element

        :param component: Either the dataframe of a component as pandas.Series or the name of the component
        :param element_name: The name of the element
        :param year: year of which full time series is selected
        :param scenario: The scenario for with the component should be extracted (only if needed)
        :param is_dual: if component is dual variable
        :param discount_to_first_step: apply annuity to first year of interval or entire interval
        :param node: Filter the results by a specific node
        :param start_time_step: Filter the results by a specific start time step
        :param end_time_step: Filter the results by a specific end time step
        :return: A dataframe containing the full timeseries of the element
        """
        # extract the data
        component_name, component_data = self._get_component_data(component, scenario, is_dual=is_dual)

        if node is not None:
            location_name = set(component_data.index.names).intersection(set(["node", "edge"])).pop()
            locations = [i for i in set(component_data.index.get_level_values(location_name)) if node in i]
            filter_index = [slice(None) for i in component_data.index.names if i not in ["node", "edge"]] + [locations]
            component_data = component_data.loc[tuple(filter_index), :]

        # loop over scenarios
        # no scenarios
        if not self.has_scenarios:
            full_ts = self._get_full_ts_for_single_scenario(
                component_data, component_name, scenario=None, element_name=element_name,
                year=year, is_dual = is_dual, discount_to_first_step=discount_to_first_step,
                start_time_step=start_time_step, end_time_step=end_time_step)
        # specific scenario
        elif scenario is not None:
            full_ts = self._get_full_ts_for_single_scenario(
                component_data, component_name, scenario=scenario, element_name=element_name,
                year=year, is_dual = is_dual, discount_to_first_step=discount_to_first_step,
                start_time_step=start_time_step, end_time_step=end_time_step)
        # all scenarios
        else:
            full_ts_dict = {}
            for scenario in self.scenarios:
                component_data_scenario = component_data[scenario]
                full_ts_dict[scenario] = self._get_full_ts_for_single_scenario(
                    component_data_scenario, component_name, scenario=scenario, element_name=element_name,
                    year=year, is_dual = is_dual,discount_to_first_step=discount_to_first_step,
                    start_time_step=start_time_step, end_time_step=end_time_step)
            if isinstance(full_ts_dict[scenario], pd.Series):
                full_ts = pd.concat(full_ts_dict, keys=full_ts_dict.keys(), axis=1).T
            else:
                full_ts = pd.concat(full_ts_dict, keys=full_ts_dict.keys())
        return full_ts

    def _get_full_ts_for_single_scenario(
            self,
            component_data,component_name,
            scenario,
            element_name=None,
            year=None,
            is_dual=False,
            discount_to_first_step=True,
            start_time_step = 0,
            end_time_step = None):
        """ calculates total value for single scenario
        :param component_data: numerical data of component
        :param component_name: name of component
        :param scenario: The scenario to calculate the total value for
        :param element_name: The element name to calculate the value for, defaults to all elements
        :param year: The year to calculate the value for, defaults to all years
        :param is_dual: if component is dual variable
        :param discount_to_first_step: apply annuity to first year of interval or entire interval
        :param start_time_step: Filter the results by a specific start time step
        :param end_time_step: Filter the results by a specific end time step
        :return: A dataframe containing the total value with the specified parameters
        """
        # timestep dict
        sequence_time_steps_dicts = self.results[scenario]["sequence_time_steps_dicts"]
        ts_type = self._get_ts_type(component_data, component_name, scenario=scenario)
        if is_dual:
            annuity = self._get_annuity(discount_to_first_step, scenario=scenario)
        else:
            annuity = pd.Series(index=self.get_years(scenario), data=1)
        if isinstance(component_data, pd.Series):
            return component_data / annuity
        if ts_type == "yearly":
            if element_name is not None:
                component_data = component_data.loc[element_name]
            component_data = component_data.div(annuity, axis=1)
            # component indexed by yearly component
            if year is not None:
                if year in component_data.columns:
                    return component_data[year]
                else:
                    print(
                        f"WARNING: year {year} not in years {component_data.columns}. Return component values for all years")
                    return component_data
            else:
                return component_data
        elif ts_type == "operational":
            is_storage = False
        else:
            is_storage = True
        time_step_duration = self._get_ts_duration(scenario, is_storage=is_storage)
        if element_name is not None:
            component_data = component_data.loc[element_name]
        # expand time steps
        if is_storage:
            sequence_time_steps = sequence_time_steps_dicts.get_sequence_time_steps(time_step_type="storage")
        else:
            sequence_time_steps = sequence_time_steps_dicts.get_sequence_time_steps(time_step_type="operation")
        # if dual variables, divide by time step operational duration
        if is_dual:
            component_data = component_data.div(time_step_duration, axis=1)
            for year_temp in annuity.index:
                time_steps_year = self._get_time_steps_of_year(sequence_time_steps_dicts,ts_type,year_temp)
                component_data[time_steps_year] = component_data[time_steps_year] / annuity[year_temp]
        # throw together
        if end_time_step is None:
            end_time_step = len(sequence_time_steps)
        sequence_time_steps = sequence_time_steps[start_time_step:end_time_step]
        sequence_time_steps = sequence_time_steps[np.in1d(sequence_time_steps, list(component_data.columns))]
        output_df = component_data[sequence_time_steps]  # .reset_index(drop=True,axis=1)
        output_df = output_df.T.reset_index(drop=True).T
        # select single year
        if year is not None:
            if year in self.get_years(scenario):
                hours_of_year = self._get_hours_of_year(year, scenario)
                output_df = (output_df[hours_of_year]).T.reset_index(drop=True).T
            else:
                print(f"WARNING: year {year} not in years {self.results[scenario]['years']}. Return component values for all years")

        return output_df

    def get_total(self, component, element_name=None, year=None, scenario=None):
        """Calculates the total Value of a component

        :param component: Either a dataframe as returned from <get_df> or the name of the component
        :param element_name: The element name to calculate the value for, defaults to all elements
        :param year: The year to calculate the value for, defaults to all years
        :param scenario: The scenario to calculate the total value for
        :return: A dataframe containing the total value with the specified parameters
        """
        # extract the data
        component_name, component_data = self._get_component_data(component, scenario)
        # loop over scenarios
        # no scenarios
        if not self.has_scenarios:
            total_value = self._get_total_for_single_scenario(
                component_data, component_name, scenario=None,
                element_name=element_name, year=year)
        # specific scenario
        elif scenario is not None:
            total_value = self._get_total_for_single_scenario(
                component_data, component_name, scenario=scenario,
                element_name=element_name, year=year)
        # all scenarios
        else:
            total_value_dict = {}
            for scenario in self.scenarios:
                component_data_scenario = component_data[scenario]
                total_value_dict[scenario] = self._get_total_for_single_scenario(
                    component_data_scenario, component_name, scenario=scenario,
                    element_name=element_name, year=year)
            if isinstance(total_value_dict[scenario], pd.Series):
                total_value = pd.concat(total_value_dict, keys=total_value_dict.keys(), axis=1).T
            else:
                total_value = pd.concat(total_value_dict, keys=total_value_dict.keys())
        return total_value

    def _get_total_for_single_scenario(self, component_data, component_name, scenario, element_name=None, year=None, split_years=True):
        """ calculates total value for single scenario
        :param component_data: numerical data of component
        :param component_name: name of component
        :param scenario: The scenario to calculate the total value for
        :param element_name: The element name to calculate the value for, defaults to all elements
        :param year: The year to calculate the value for, defaults to all years
        :param split_years: Calculate the value for each year individually
        :return: A dataframe containing the total value with the specified parameters
        """
        sequence_time_steps_dicts = self.results[scenario]["sequence_time_steps_dicts"]
        if isinstance(component_data, pd.Series):
            return component_data
        ts_type = self._get_ts_type(component_data, component_name, scenario=scenario, force_output=True)
        if ts_type is None:
            return component_data
        elif ts_type == "yearly":
            if element_name is not None:
                component_data = component_data.loc[element_name]
            if year is not None:
                if year in component_data.columns:
                    return component_data[year]
                else:
                    print(
                        f"WARNING: year {year} not in years {component_data.columns}. Return total value for all years")
                    return component_data.sum(axis=1)
            else:
                return component_data
        elif ts_type == "operational":
            is_storage = False
        else:
            is_storage = True

        # extract time step duration
        time_step_duration = self._get_ts_duration(scenario, is_storage=is_storage)

        # If we have an element name
        if element_name is not None:
            # check that it is in the index
            assert element_name in component_data.index.get_level_values(
                level=0), f"element {element_name} is not found in index of {component_name}"
            # get the index
            component_data = component_data.loc[element_name]

        total_value = component_data.multiply(time_step_duration,axis=1)
        if year is not None:
            time_steps_year = self._get_time_steps_of_year(sequence_time_steps_dicts,ts_type,year)
            total_value = total_value[time_steps_year].sum(axis=1)
        else:
            total_value_temp = pd.DataFrame(index=total_value.index, columns=self.get_years(scenario))
            for year_temp in self.get_years(scenario):
                time_steps_year = self._get_time_steps_of_year(sequence_time_steps_dicts,ts_type,year_temp)
                total_value_temp[year_temp] = total_value[time_steps_year].sum(axis=1)
            total_value = total_value_temp
        return total_value

    def get_dual(self, constraint, scenario=None, element_name=None, year=None, discount_to_first_step=True):
        """ extracts the dual variables of a constraint

        :param constraint: #TODO describe parameter/return
        :param scenario: #TODO describe parameter/return
        :param element_name: #TODO describe parameter/return
        :param year: #TODO describe parameter/return
        :param discount_to_first_step: apply annuity to first year of interval or entire interval
        :return: #TODO describe parameter/return
        """
        if not self.results[scenario]["solver"]["add_duals"]:
            logging.warning("Duals are not calculated. Skip.")
            return
        _duals = self.get_full_ts(component=constraint, scenario=scenario, is_dual=True, element_name=element_name, year=year, discount_to_first_step=discount_to_first_step)
        return _duals

    def get_doc(self, component, index_set=False):
        """extracts the doc string of a component

        :param component: name of parameter/variable/constraint
        :param index_set: bool to choose whether component's index_set should be returned as well
        :return: docstring of component
        """
        scenario = None
        if None not in self.scenarios:
            scenario = "scenario_"
        strings = self.results[scenario][None]["pars_and_vars"][component]["docstring"]
        string_list = strings.split(";")
        doc = [doc for doc in string_list if "doc" in doc]
        if index_set:
            index_set = [ind_set for ind_set in string_list if "dims" in ind_set]
            return doc, index_set
        return doc[0]

    def get_system(self,scenario=None):
        """ returns the system of a model. For multiple scenarios, return all
        :param scenario: scenario in model
        :return: system dict """
        system = self._get_config_dict(config_type="system",scenario=scenario)
        return system

    def get_analysis(self,scenario=None):
        """ returns the analysis of a model. For multiple scenarios, return all
        :param scenario: scenario in model
        :return: analysis dict """
        analysis = self._get_config_dict(config_type="analysis",scenario=scenario)
        return analysis

    def get_solver(self,scenario=None):
        """ returns the solver of a model. For multiple scenarios, return all
        :param scenario: scenario in model
        :return: solver dict """
        solver = self._get_config_dict(config_type="solver",scenario=scenario)
        return solver

    def get_years(self,scenario=None):
        """ returns the optimization years of a model.
        If a model has scenarios and no scenario is specified, use first one.
        :param scenario: scenario in model
        :return: years"""
        if self.has_scenarios and scenario is None:
            scenario = self.scenarios[0]
        system = self.get_system(scenario)
        years = list(range(0, system["optimized_years"]))
        return years

    def has_MF(self,scenario=None):
        """ returns if the model is myopic foresight .
        If a model has scenarios and no scenario is specified, use first one.
        :param scenario: scenario in model
        :return: boolean if model has myopic foresight"""
        if self.has_scenarios and scenario is None:
            scenario = self.scenarios[0]
        has_mf = self._get_config_dict(config_type="has_MF",scenario=scenario)
        return has_mf

    def _get_config_dict(self, config_type,scenario=None):
        """ returns a config dict (system, analysis, solver) of a model. For multiple scenarios, return all
        :param config_type: name of config type (system, analysis, solver)
        :param scenario: manual scenario
        :return: config of model """
        # has scenarios
        if self.has_scenarios:
            if scenario is not None:
                if scenario in self.scenarios:
                    config = self.results[scenario][config_type]
                    return config
                else:
                    logging.warning(f"Requested scenario {scenario} not in {self.scenarios} of {self.name}")
            config = {}
            for s in self.scenarios:
                config[s] = self.results[s][config_type]
        else:
            config = self.results[None][config_type]
        return config

    def _get_annuity(self,discount_to_first_step, scenario):
        """ discounts the duals

        :param discount_to_first_step: apply annuity to first year of interval or entire interval
        :param scenario: scenario name whose results are assessed
        :return: #TODO describe parameter/return
        """
        system = self.results[scenario]["system"]
        # calculate annuity
        discount_rate = self.get_df("discount_rate").squeeze()
        annuity = pd.Series(index=self.get_years(scenario), dtype=float)
        for year in self.get_years(scenario):
            interval_between_years = system["interval_between_years"]
            if year == self.get_years(scenario)[-1]:
                interval_between_years_this_year = 1
            else:
                interval_between_years_this_year = system["interval_between_years"]
            if self.results[scenario]["has_MF"]:
                if discount_to_first_step:
                    annuity[year] = interval_between_years_this_year * (1 / (1 + discount_rate))
                else:
                    annuity[year] = sum(((1 / (1 + discount_rate)) ** (_intermediate_time_step)) for _intermediate_time_step in range(0, interval_between_years_this_year))
            else:
                if discount_to_first_step:
                    annuity[year] = interval_between_years_this_year*((1 / (1 + discount_rate)) ** (interval_between_years * (year - self.get_years(scenario)[0])))
                else:
                    annuity[year] = sum(((1 / (1 + discount_rate)) ** (interval_between_years * (year - self.get_years(scenario)[0]) + _intermediate_time_step))
                        for _intermediate_time_step in range(0, interval_between_years_this_year))
        return annuity

    def _get_ts_duration(self, scenario=None, is_storage=False):
        """ extracts the time steps duration

        :param scenario: #TODO describe parameter/return
        :param is_storage: #TODO describe parameter/return
        :return: #TODO describe parameter/return
        """
        # extract the right timestep duration
        if self.has_scenarios:
            if scenario is None:
                raise ValueError("Please specify a scenario!")
            else:
                if is_storage:
                    time_step_duration = self.time_step_storage_duration[scenario]
                else:
                    time_step_duration = self.time_step_operational_duration[scenario]
        else:
            if is_storage:
                time_step_duration = self.time_step_storage_duration
            else:
                time_step_duration = self.time_step_operational_duration
        return time_step_duration

    def _get_component_data(self, component, scenario=None, is_dual=False):
        """ extracts the data for a component

        :param component: #TODO describe parameter/return
        :param scenario: #TODO describe parameter/return
        :param is_dual: #TODO describe parameter/return
        :return: #TODO describe parameter/return
        """
        # extract the data
        if isinstance(component, str):
            component_name = component
            # only use the data from one scenario if specified
            if scenario is not None:
                component_data = self.get_df(component, is_dual=is_dual)[scenario]
            else:
                component_data = self.get_df(component, is_dual=is_dual)
            if isinstance(component_data, dict):
                component_data_temp = {}
                for key,data in component_data.items():
                    if isinstance(data.index, pd.MultiIndex):
                        component_data_temp[key] = self._unstack_time_level(data, component_name, scenario=scenario)
                    else:
                        component_data_temp[key] = data
                component_data = component_data_temp
            elif isinstance(component_data.index, pd.MultiIndex):
                component_data = self._unstack_time_level(component_data, component_name, scenario)
        elif isinstance(component, pd.Series):
            component_name = component.name
            component_data = self._unstack_time_level(component, component_name, scenario=scenario)
        else:
            raise TypeError(f"Type {type(component).__name__} of input is not supported.")

        return component_name, component_data

    def _get_ts_type(self, component_data, component_name, scenario, force_output=False):
        """ get time step type (operational, storage, yearly)

        :param component_data: #TODO describe parameter/return
        :param component_name: #TODO describe parameter/return
        :param scenario: scenario name whose ts type is examined
        :param force_output: #TODO describe parameter/return
        :return: #TODO describe parameter/return
        """
        header_operational = self.results[scenario]["analysis"]["header_data_inputs"]["set_time_steps_operation"]
        header_storage = self.results[scenario]["analysis"]["header_data_inputs"]["set_time_steps_storage_level"]
        header_yearly = self.results[scenario]["analysis"]["header_data_inputs"]["set_time_steps_yearly"]
        if isinstance(component_data, pd.Series):
            axis_name = component_data.index.name
        else:
            axis_name = component_data.columns.name
        if axis_name == header_operational:
            return "operational"
        elif axis_name == header_storage:
            return "storage"
        elif axis_name == header_yearly:
            return "yearly"
        else:
            if force_output:
                return None
            else:
                raise KeyError(f"Axis index name of '{component_name}' ({axis_name}) is unknown. Should be (operational, storage, yearly)")

    def _get_time_steps_of_year(self,sequence_time_steps_dicts,ts_type,year):
        """
        returns time steps of given year
        :param sequence_time_steps_dicts:
        :param ts_type:
        :param year:
        :return:
        """
        if ts_type != "storage" and ts_type != "operational":
            raise KeyError(f"Time step type {ts_type} unknown.")
        if not (isinstance(sequence_time_steps_dicts,TimeStepDictFromFile) and not self.new_time_steps):
            if ts_type == "storage":
                time_steps_year = sequence_time_steps_dicts.get_time_steps_year2storage(year)
            else:
                time_steps_year = sequence_time_steps_dicts.get_time_steps_year2operation(year)
        else:
            if self.has_scenarios:
                tech_proxy = self.get_system(scenario=self.scenarios[0])["set_storage_technologies"][0]
            else:
                tech_proxy = self.get_system()["set_storage_technologies"][0]
            if ts_type == "storage":
                 tech_proxy = tech_proxy + "_storage_level"
            time_steps_year = sequence_time_steps_dicts.get_time_steps_year2operation_old(tech_proxy, year)
        return time_steps_year

    def _unstack_time_level(self, component, component_name, scenario):
        """ unstacks the time level of a dataframe

        :param component: pd.Series of component
        :param component_name: name of component
        :returns unstacked_component: pd.Dataframe of unstacked component
        """
        headers = []
        if scenario is None:
            scenario = self.scenarios[0]
        headers.append(self.results[scenario]["analysis"]["header_data_inputs"]["set_time_steps_operation"])
        headers.append(self.results[scenario]["analysis"]["header_data_inputs"]["set_time_steps_yearly"])
        if self.new_time_steps:
            headers.append(self.results[scenario]["analysis"]["header_data_inputs"]["set_time_steps_storage"])
        else:
            headers.append(self.results[scenario]["analysis"]["header_data_inputs"]["set_time_steps_storage_level"])
        headers = set(headers)
        sel_header = list(headers.intersection(component.index.names))
        assert len(sel_header) <= 1, f"Index of component {component_name} has multiple time headers: {sel_header}"
        if len(sel_header) == 1:
            unstacked_component = component.unstack(sel_header)
        else:
            unstacked_component = component
        return unstacked_component

    def _get_hours_of_year(self, year, scenario):
        """ get total hours of year

        :param year: #TODO describe parameter/return
        :param scenario: scenario name whose hours of specific year are accessed
        :return: #TODO describe parameter/return
        """
        _total_hours_per_year = self.results[scenario]["system"]["unaggregated_time_steps_per_year"]
        _hours_of_year = list(range(year * _total_hours_per_year, (year + 1) * _total_hours_per_year))
        return _hours_of_year

    def __str__(self):
        return f"Results of '{self.path}'"

    def standard_plots(self, save_fig=False, file_type=None):
        """Plots data of basic variables to get a first overview of the results

        :param save_fig: Choose if figure should be saved as pdf
        :param file_type: File type the figure is saved as (pdf, svg, png, ...)
        """
        scenario = None
        if None not in self.scenarios:
            scenario = "scenario_"
        demand = self.get_df("demand", scenario=scenario)
        #remove carriers without demand
        demand = demand.loc[(demand != 0), :]
        for carrier in demand.index.levels[0].values:
            if carrier in demand:
                self.plot("capacity", yearly=True, tech_type="conversion", reference_carrier=carrier, plot_strings={"title": f"Capacities of {carrier.capitalize()} Generating Conversion Technologies", "ylabel": "Capacity"}, save_fig=save_fig, file_type=file_type)
                self.plot("capacity_addition", yearly=True, tech_type="conversion", reference_carrier=carrier, plot_strings={"title": f"Capacity Addition of {carrier.capitalize()} Generating Conversion Technologies", "ylabel": "Capacity"}, save_fig=save_fig, file_type=file_type)
                self.plot("flow_conversion_input", yearly=True, reference_carrier=carrier, plot_strings={"title": f"Input Flows of {carrier.capitalize()} Generating Conversion Technologies", "ylabel": "Input Flow"}, save_fig=save_fig, file_type=file_type)
                self.plot("flow_conversion_output", yearly=True, reference_carrier=carrier, plot_strings={"title": f"Output Flows of {carrier.capitalize()} Generating Conversion Technologies", "ylabel": "Output Flow"}, save_fig=save_fig, file_type=file_type)
        self.plot("cost_capex_total",yearly=True, plot_strings={"title": "Total Capex", "ylabel": "Capex"}, save_fig=save_fig, file_type=file_type)
        self.plot("cost_opex_total", yearly=True, plot_strings={"title": "Total Opex", "ylabel": "Opex"}, save_fig=save_fig, file_type=file_type)
        self.plot("cost_carrier", yearly=True, plot_strings={"title": "Carrier Cost", "ylabel": "Cost"}, save_fig=save_fig, file_type=file_type)
        self.plot("cost_carbon_emissions_total", yearly=True, plot_strings={"title": "Total Carbon Emissions Cost", "ylabel": "Cost"}, save_fig=save_fig, file_type=file_type)
        #plot total costs as stacked bar plot of individual costs
        costs = ["cost_capex_total", "cost_opex_total", "cost_carbon_emissions_total", "cost_carrier_total", "cost_shed_demand"]
        total_cost = pd.DataFrame()
        for cost in costs:
            test = self.get_total(cost, scenario=scenario)
            if cost == "cost_shed_demand":
                cost_df = self.get_total(cost, scenario=scenario).sum(axis=0)
                cost_df.name = "cost_shed_demand_total"
                total_cost = pd.concat([total_cost, cost_df], axis=1)
            else:
                total_cost = pd.concat([total_cost, self.get_total(cost, scenario=scenario)], axis=1)
        self.plot(total_cost.transpose(), yearly=True, node_edit="all" ,plot_strings={"title": "Total Cost", "ylabel": "Cost"}, save_fig=save_fig, file_type=file_type)

    def get_energy_balance_df(
            self,
            node,
            carrier,
            scenario=None,
            start_time_step = 0,
            end_time_step = None
        ) -> pd.DataFrame:
        components = ["flow_conversion_output", "flow_conversion_input", "flow_export", "flow_import", "flow_storage_charge", "flow_storage_discharge", "demand", "flow_transport_in", "flow_transport_out", "shed_demand"]
        lowers = ["flow_conversion_input", "flow_export", "flow_storage_charge", "flow_transport_out"]

        data_plot = pd.DataFrame()

        if scenario not in self.scenarios:
            scenario = "scenario_"
        for component in components:
            if component == "flow_transport_in":
                full_ts = self.get_full_ts("flow_transport", scenario=scenario, node=node, start_time_step=start_time_step, end_time_step=end_time_step)
                data_full_ts = self.edit_carrier_flows(full_ts, node, carrier, "in", scenario)
            elif component == "flow_transport_out":
                full_ts = self.get_full_ts("flow_transport", scenario=scenario, node=node, start_time_step=start_time_step, end_time_step=end_time_step)
                data_full_ts = self.edit_carrier_flows(full_ts, node, carrier, "out", scenario)
            else:
                # get full timeseries of component and extract rows of relevant node
                full_ts = self.get_full_ts(component, scenario=scenario, node=node, start_time_step=start_time_step, end_time_step=end_time_step)
                data_full_ts = self.edit_nodes(full_ts, node, scenario)
                # extract data of desired carrier
                data_full_ts = self.extract_carrier(data_full_ts, carrier, scenario)
                # check if return from extract_carrier() is still a data frame as it is possible that the desired carrier isn't contained --> None returned
                if not isinstance(data_full_ts, pd.DataFrame):
                    continue
            # change sign of variables which enter the node
            if component in lowers:
                data_full_ts = data_full_ts.multiply(-1)
            if isinstance(data_full_ts, pd.DataFrame):
                # add variable name to multi-index such that they can be shown in plot legend
                data_full_ts = pd.concat([data_full_ts], keys=[component], names=["variable"])
                # drop unnecessary index levels to improve the plot legend's appearance
                if data_full_ts.index.nlevels == 3:
                    data_full_ts = data_full_ts.droplevel([2])
                elif data_full_ts.index.nlevels == 4:
                    data_full_ts = data_full_ts.droplevel([2,3])
                # transpose data frame as pandas plot function plots column-wise
                data_full_ts = data_full_ts.transpose()
            elif isinstance(data_full_ts, pd.Series):
                data_full_ts.name = component
            # add data of current variable to the plot data frame
            data_plot = pd.concat([data_plot, data_full_ts], axis=1)
        return data_plot

    def plot_energy_balance(self, node, carrier, year, start_hour=None, duration=None, save_fig=False, file_type=None, demand_area=False, scenario=None):
        """Visualizes the energy balance of a specific carrier at a single node

        :param node: node of interest
        :param carrier: String of carrier of interest
        :param year: Generic index of year of interest
        :param start_hour: Specific hour of year, where plot should start (needs to be passed together with duration)
        :param duration: Number of hours that should be plotted from start_hour
        :param save_fig: Choose if figure should be saved as pdf
        :param file_type: File type the figure is saved as (pdf, svg, png, ...)
        :param demand_area: Choose if carrier demand should be plotted as area with other negative flows (instead of line)
        :param scenario: Choose scenario of interest (only for multi-scenario simulations, per default scenario_ is plotted)
        """
        # plt.rcParams["figure.figsize"] = (30*1, 6.5*1)
        fig,ax = plt.subplots(figsize=(30,6.5),layout = "constrained")
        # extract the rows of the desired year
        fetch_fast = True

        if scenario not in self.scenarios:
            scenario = self.scenarios[0]

        if year not in list(range(self.results[scenario]["system"]["optimized_years"])):
            warnings.warn(f"Chosen year '{year}' has not been optimized")
            data_plot = self.get_energy_balance_df(node, carrier, scenario)
        else:
            ts_per_year = self.results[scenario]["system"]["unaggregated_time_steps_per_year"]
            if fetch_fast:
                data_plot = self.get_energy_balance_df(node, carrier, scenario, start_time_step=ts_per_year*year, end_time_step=ts_per_year*year+ts_per_year)
            else:
                data_plot = self.get_energy_balance_df(node, carrier, scenario)
                data_plot = data_plot.iloc[ts_per_year*year:ts_per_year*year+ts_per_year]

        # extract specific hours of year
        data_plot = data_plot.reset_index(drop=True)
        if start_hour is not None and duration is not None:
            data_plot = data_plot.iloc[start_hour:start_hour+duration]
        # remove columns(technologies/variables) with constant zero value
        data_plot = data_plot.loc[:, (data_plot != 0).any(axis=0)]
        # set colors and plot data frame, repeat tab20
        num_repeat = np.ceil(data_plot.shape[1]/20)
        colors = np.tile(np.array(plt.cm.tab20.colors),(int(num_repeat),1))
        # check if demand should be plotted as a line or as an area
        if demand_area is False:
            data_plot_wo_demand = data_plot.drop(columns=[demand for demand in data_plot.columns if "demand" in demand or "shed_demand" in demand])
            data_plot_wo_demand.plot(kind="area",ax=ax, stacked=True, alpha=1, color=colors, title="Energy Balance " + carrier + " " + node + " " + str(year), ylabel="Power", xlabel="Time", legend=False)
            data_plot = data_plot[[col for col in data_plot.columns if 'demand' in col or "shed_demand" in col]]
            # check if there is demand for the chosen carrier at all
            if not data_plot.empty:
                # plot demand (without shed demand)
                data_plot[data_plot.columns[0]].plot(kind="line", ax=ax, label='demand', color="black", legend=False)
                # check if there is shed demand
                if data_plot.shape[1] == 2:
                    # compute served demand as demand minus shed demand
                    data_demand_minus_shed_demand = pd.DataFrame({"served_demand": data_plot[data_plot.columns[0]] - data_plot[data_plot.columns[1]]})
                    data_demand_minus_shed_demand.plot(kind="line", ax=ax, label="served_demand", color="k", linestyle="--", legend=False)

                    # as the demand line can "leave" the pyplot figure when there is demand shedding, the figure's y-limits need to be adjusted manually
                    max_value_wo_demand = data_plot_wo_demand.clip(lower=0).sum(axis=1).max()
                    min_value_wo_demand = data_plot_wo_demand.clip(upper=0).sum(axis=1).min()
                    # find the overall minimal and maximal values of either the stacked areas or the demand curve
                    min_value = min(min_value_wo_demand, data_plot.values.min())
                    max_value = max(max_value_wo_demand, data_plot.values.max())
                    # adjust the figure's y-limits accordingly
                    plt.ylim(min_value*1.05, max_value*1.05)
        else:
            if "demand" in data_plot.columns and "shed_demand" not in data_plot.columns:
                data_plot["demand"] = data_plot["demand"].multiply(-1)
                data_plot.plot(kind="area", ax=ax, stacked=True, color=colors, title="Energy Balance " + carrier + " " + node + " " + str(year), ylabel="Power", xlabel="Time", legend=False)
            elif "demand" in data_plot.columns and "shed_demand" in data_plot.columns:
                data_plot["served_demand"] = data_plot["demand"] - data_plot["shed_demand"]
                data_plot = data_plot.drop(columns=["demand"])
                data_plot["served_demand"] = data_plot["served_demand"].multiply(-1)
                data_plot["shed_demand"] = data_plot["shed_demand"].multiply(-1)
                # extract all column names as a list
                all_columns = data_plot.columns.tolist()
                # change the order of the served_demand and the shed_demand such that the plot shows an improved appearance
                data_plot = data_plot.reindex(columns=all_columns[0:-2] + [all_columns[-1], all_columns[-2]])
                data_plot.plot(kind="area", ax=ax, stacked=True, color=colors, title="Energy Balance " + carrier + " " + node + " " + str(year), ylabel="Power", xlabel="Time", legend=False)
            else:
                data_plot.plot(kind="area", ax=ax, stacked=True, color=colors, title="Energy Balance " + carrier + " " + node + " " + str(year), ylabel="Power", xlabel="Time", legend=False)
        # xlim
        if start_hour is None and duration is None:
            ax.set_xlim([0, data_plot.shape[0]-1])
        else:
            ax.set_xlim([start_hour, start_hour+duration-1])
        # legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside right')
        if save_fig:
            path = os.path.join(os.getcwd(), "outputs")
            path = os.path.join(path, os.path.basename(self.results[scenario]["analysis"]["dataset"]))
            path = os.path.join(path, "result_plots")
            if not os.path.exists(path):
                os.makedirs(path)
            if file_type in plt.gcf().canvas.get_supported_filetypes():
                if start_hour is None and duration is None:
                    plt.savefig(os.path.join(path, "energy_balance_" + carrier + "_" + node + "_" + str(year) + "." + file_type))
                else:
                    plt.savefig(os.path.join(path, "energy_balance_" + carrier + "_" + node + "_" + str(year) + "_" + str(start_hour) + "_" + str(duration) + "." + file_type))
            elif file_type is None:
                #save figure as pdf if file_type has not been specified
                if start_hour is None and duration is None:
                    plt.savefig(os.path.join(path, "energy_balance_" + carrier + "_" + node + "_" + str(year) + ".pdf"))
                else:
                    plt.savefig(os.path.join(path, "energy_balance_" + carrier + "_" + node + "_" + str(year) + "_" + str(start_hour) + "_" + str(duration) + ".pdf"))
            else:
                warnings.warn(f"Plot couldn't be saved as specified file type '{file_type}' isn't supported or has not been passed in the following form: 'pdf', 'svg', 'png', etc.")
        plt.show()

    def plot(self, component, yearly=False, node_edit=None, sum_techs=False, tech_type=None, plot_type=None, reference_carrier=None, plot_strings={"title": "", "ylabel": ""}, save_fig=False, file_type=None, year=None, start_hour=None, duration=None, scenario=None):
        """Plots component data as specified by arguments

        :param component: Either the name of the component or a data frame of the component's data
        :param yearly: Operational time steps if false, else yearly time steps
        :param node_edit: By default, data is summed over nodes, chose node_edit="all" or a specific node (e.g. node_edit="CH") such that data is not summed over nodes or that a single node's data is extracted, respectively
        :param sum_techs: sum values of technologies per carrier if true
        :param tech_type: specify whether transport, storage or conversion technologies should be plotted separately (useful for capacity, etc.)
        :param plot_type: per default stacked bar plot, passing bar will plot normal bar plot
        :param reference_carrier: specify reference carrier such as electricity, heat, etc. to extract their data
        :param plot_strings: Dict of strings used to set title and labels of plot
        :param save_fig: Choose if figure should be saved as
        :param file_type: File type the figure is saved as (pdf, svg, png, ...)
        :param year: Year of interest (only for operational plots)
        :param start_hour: Specific hour of year, where plot should start (needs to be passed together with duration) (only for operational plots)
        :param duration: Number of hours that should be plotted from start_hour (only for operational plots)
        :param scenario: Choose scenario of interest (only for multi-scenario simulations, per default scenario_ is plotted)
        :return: plot
        """
        if isinstance(component, str):
            if None in self.scenarios:
                if component not in self.results[None][None]["pars_and_vars"]:
                    warnings.warn(f"Chosen component '{component}' doesn't exist")
                    return
            else:
                if scenario is None:
                    scenario = self.scenarios[0]
                if component not in self.results[scenario][None]["pars_and_vars"]:
                    warnings.warn(f"Chosen component '{component}' doesn't exist")
                    return

            component_name, component_data = self._get_component_data(component, scenario=scenario)
        # set timeseries type
        if yearly:
            ts_type = "yearly"
        else:
            ts_type = self._get_ts_type(component_data, component_name, scenario=scenario)

        # needed for plot titles
        title = f"{component}, "
        arguments = [node_edit, sum_techs, tech_type, reference_carrier, year, start_hour, duration, scenario]
        argument_names = ["node_edit", "sum_techs", "tech_type", "reference_carrier", "year", "start_hour", "duration", "scenario"]

        # plot variable with operational time steps
        if ts_type == "operational":
            if isinstance(component, str):
                data_full_ts = self.get_full_ts(component, scenario=scenario)
            elif isinstance(component, pd.DataFrame):
                data_full_ts = component
            # extract desired data
            if node_edit != "all":
                data_full_ts = self.edit_nodes(data_full_ts, node_edit, scenario=scenario)
            if sum_techs:
                data_full_ts = self.sum_over_technologies(data_full_ts)
            if reference_carrier != None:
                data_full_ts = self.extract_reference_carrier(data_full_ts, reference_carrier, scenario)
            # drop index levels having constant value in all indices
            if isinstance(data_full_ts, pd.DataFrame) and data_full_ts.index.nlevels > 1:
                drop_levs = []
                for ind, lev_shape in enumerate(data_full_ts.index.levshape):
                    if ind != 0:
                        if all(lev_value[ind] == data_full_ts.index.values[0][ind] for lev_value in data_full_ts.index.values[ind:]):
                            drop_levs.append(ind)
                data_full_ts = data_full_ts.droplevel(level=drop_levs)
            # extract data of a specific year
            data_full_ts = data_full_ts.transpose()
            if year is not None:
                # extract the rows of the desired year
                ts_per_year = self.results[scenario]["system"]["unaggregated_time_steps_per_year"]
                data_full_ts = data_full_ts.iloc[ts_per_year*year:ts_per_year*year+ts_per_year]
                # extract specific hours of year
                if start_hour is not None and duration is not None:
                    data_full_ts = data_full_ts.iloc[start_hour:start_hour + duration]
            # remove columns(technologies/variables) with constant zero value
            data_full_ts = data_full_ts.loc[:, (data_full_ts != 0).any(axis=0)]
            colors = plt.cm.tab20(range(data_full_ts.shape[1]))
            plt.rcParams["figure.figsize"] = (30 * 1, 6.5 * 1)
            # create title containing argument values
            for ind, arg in enumerate(arguments):
                if arg is not None and arg is not False:
                    title += argument_names[ind]+": "+ f"{arg}, "
            # check if there is a title passed by function argument
            if plot_strings["title"] != "":
                title = plot_strings["title"]
            data_full_ts.plot(kind="area", stacked=True, color=colors, title=title)
            plt.xlabel("Time [hours]")
            plt.ylabel(component)

        # plot variable with yearly time steps
        elif ts_type == "yearly":
            if isinstance(component, str):
                data_total = self.get_total(component, scenario=scenario)
            elif isinstance(component, pd.DataFrame):
                data_total = component
            if tech_type != None:
                data_total = self.extract_technology(data_total, tech_type, scenario=scenario)
            if reference_carrier != None:
                data_total = self.extract_reference_carrier(data_total, reference_carrier, scenario)
            # sum data according to chosen options
            if node_edit != "all":
                data_total = self.edit_nodes(data_total, node_edit, scenario=scenario)
            if sum_techs:
                data_total = self.sum_over_technologies(data_total)
            #drop index levels having constant value in all indices
            if isinstance(data_total, pd.DataFrame) and data_total.index.nlevels > 1:
                drop_levs = []
                for ind, lev_shape in enumerate(data_total.index.levshape):
                    if ind != 0:
                        if all(lev_value[ind] == data_total.index.values[0][ind] for lev_value in data_total.index.values):
                            drop_levs.append(ind)
                data_total = data_total.droplevel(level=drop_levs)

            #create title containing argument values
            for ind, arg in enumerate(arguments):
                if arg is not None and arg is not False:
                    title += argument_names[ind]+": "+ f"{arg}, "
            #check if there is a title passed by function argument
            if plot_strings["title"] != "":
                title = plot_strings["title"]

            if plot_type == None:
                if isinstance(data_total, pd.DataFrame):
                    data_total = data_total.transpose()
                plt.rcParams["figure.figsize"] = (9.5, 6.5)
                fig, ax = plt.subplots()
                data_total.plot(ax=ax, kind='bar', stacked=True,
                        title=title, rot=0, xlabel='Year', ylabel=plot_strings["ylabel"])
                pos = ax.get_position()
                ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height])
                ax.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))

            elif plot_type == "bar":
                #set up bar plot
                bars = []
                for ind, row in enumerate(data_total.values):
                    bar = plt.bar(data_total.columns.values + 1/(data_total.shape[0]+1) * ind, row, 1/(data_total.shape[0]+1))
                    bars.append(bar)
                #data_total.columns.values dtype needs to be cast to an integer as edit_nodes changes the dtype to an object
                plt.xticks(np.array(data_total.columns.values, dtype=int) + 1/(data_total.shape[0]+1) * 1/2 * (data_total.shape[0]-1), np.array(data_total.columns.values, dtype=int)+self.results[scenario]["system"]["reference_year"])
                plt.legend((bars),(data_total.index.values),ncols=max(1,int(data_total.shape[0]/7)))

            else:
                warnings.warn(f"Chosen plot_type '{plot_type}' doesn't exist, chose 'bar' or don't pass any argument for stacked bar plot")

        #plot variable with storage time steps
        elif ts_type == "storage":
            warnings.warn("Further implementation needed")

        if save_fig:
            path = os.path.join(os.getcwd(), "outputs")
            path = os.path.join(path, os.path.basename(self.results[scenario]["analysis"]["dataset"]))
            path = os.path.join(path, "result_plots")
            if not os.path.exists(path):
                os.makedirs(path)
            if file_type in plt.gcf().canvas.get_supported_filetypes():
                if isinstance(component,str):
                    plt.savefig(os.path.join(path,component + "_yearly="+str(yearly) + "_" + "node_edit=" + str(node_edit) +"." + file_type))
                else:
                    plt.savefig(os.path.join(path, plot_strings["title"] + "_yearly=" + str(yearly) + "_" + "node_edit=" + str(node_edit) + "." + file_type))
            elif file_type is None:
                #save figure as pdf if file_type has not been specified
                if isinstance(component, str):
                    plt.savefig(os.path.join(path, component + "_yearly=" + str(yearly) + "_" + "node_edit=" + str(node_edit) + ".pdf" ))
                else:
                    plt.savefig(os.path.join(path, plot_strings["title"] + "_yearly=" + str(yearly) + "_" + "node_edit=" + str(node_edit) + ".pdf"))
            else:
                warnings.warn(f"Plot couldn't be saved as specified file type '{file_type}' isn't supported or has not been passed in the following form: 'pdf', 'svg', 'png', etc.")
        plt.show()

    def calculate_connected_edges(self, node, direction: str, set_nodes_on_edges):
        """ calculates connected edges going in (direction = 'in') or going out (direction = 'out')

        :param node: current node, connected by edges
        :param direction: direction of edges, either in or out. In: node = endnode, out: node = startnode
        :param set_nodes_on_edges: set of nodes on edges
        :return set_connected_edges: list of connected edges """
        if direction == "in":
            # second entry is node into which the flow goes
            set_connected_edges = [edge for edge in set_nodes_on_edges if set_nodes_on_edges[edge][1] == node]
        elif direction == "out":
            # first entry is node out of which the flow starts
            set_connected_edges = [edge for edge in set_nodes_on_edges if set_nodes_on_edges[edge][0] == node]
        else:
            raise KeyError(f"invalid direction '{direction}'")
        return set_connected_edges

    def edit_carrier_flows(self, data, node, carrier, direction, scenario):
        """Extracts data of carrier_flow variable as needed for the plot_energy_balance function

        :param data: pd.DataFrame containing data to extract
        :param node: node of interest
        :param carrier: carrier of interest
        :param direction: flow direction with respect to node
        :param scenario: scenario of interest
        :return: pd.DataFrame containing carrier_flow data desired
        """
        set_nodes_on_edges = self.get_df("set_nodes_on_edges", is_set=True, scenario=scenario)
        set_nodes_on_edges = {edge: set_nodes_on_edges[edge].split(",") for edge in set_nodes_on_edges.index}
        data = data.loc[(slice(None), self.calculate_connected_edges(node, direction, set_nodes_on_edges)), :]

        if "carrier" not in data.index.dtypes:
            reference_carriers = self.get_df("set_reference_carriers", is_set=True, scenario=scenario)
            data_extracted = pd.DataFrame()
            data = data.groupby(["technology"]).sum()
            for ind, tech in enumerate(data.index.get_level_values("technology")):
                if reference_carriers[tech] == carrier:
                    data_extracted = pd.concat([data_extracted, data.transpose()[tech]], axis=1)

        return data_extracted.transpose()

    def edit_nodes(self, data, node_edit, scenario):
        """Manipulates the data frame 'data' as specified by 'node_edit'

        :param data: pd.DataFrame containing data to extract
        :param node_edit: string to specify if data should be summed over nodes (node_edit="all") or if a single node should be extracted (e.g. node_edit="CH")
        :param scenario: scenario name
        :return: pd.DataFrame containing data of interest
        """
        if isinstance(data, pd.Series):
            return data
        if "node" not in data.index.dtypes and "location" not in data.index.dtypes:
            return data
        # check if data of specific node, specified by string, should be extracted
        if node_edit in self.results[scenario]["system"]["set_nodes"]:
            if data.index.nlevels == 2:
                data = data.loc[(slice(None), node_edit), :]
                return data
            elif data.index.nlevels == 3:
                data = data.loc[(slice(None), slice(None), node_edit), :]
                return data
            elif data.index.nlevels == 4:
                return
        # check if data varying only in node value should be summed
        elif node_edit is None:
            level_names = [name for name in data.index.names if name not in ["node", "location"]]
            if len(level_names) == 1:
                data_summed = data.groupby([level_names[0]]).sum()
                return data_summed
            elif len(level_names) == 2:
                data_summed = data.groupby([level_names[0], level_names[1]]).sum()
                return data_summed
            else:
                warnings.warn(f"Further implementation needed")
        else:
            warnings.warn(f"Chosen node_edit string '{node_edit}' is invalid")
            return data

    def sum_over_technologies(self, data):
        """Sums the data of technologies with the same output carrier

        :param data: pd.DataFrame containing data to extract
        :return: pd.DataFrame containing data of interest
        """
        if "technology" not in data.index.dtypes:
            return data
        level_names = [name for name in data.index.names if name not in ["technology"]]
        if len(level_names) == 1:
            data_summed = data.groupby([level_names[0]]).sum()
            return data_summed
        elif len(level_names) == 2:
            data_summed = data.groupby([level_names[0], level_names[1]]).sum()
            return data_summed
        else:
            warnings.warn(f"Further implementation needed")
            return data

    def extract_carrier(self, data, carrier, scenario):
        """Extracts data of all technologies with the specified reference carrier

        :param data: pd.DataFrame containing data to extract
        :param carrier: carrier of interest
        :param scenario: scenario of interest
        :return: pd.DataFrame containing data of interest
        """
        reference_carriers = self.get_df("set_reference_carriers", is_set=True, scenario=scenario)
        if "carrier" not in data.index.dtypes:
            data_extracted = pd.DataFrame()
            for tech in data.index.get_level_values("technology"):
                if reference_carriers[tech] == carrier:
                    data_extracted = pd.concat([data_extracted, data.loc[(tech, slice(None)), :]], axis=0)
            return data_extracted
        # check if desired carrier isn't contained in data (otherwise .loc raises an error)
        if carrier not in data.index.get_level_values("carrier"):
            return None
        if data.index.nlevels == 2:
            data = data.loc[(carrier, slice(None)), :]
            return data
        elif data.index.nlevels == 3:
            data = data.loc[(slice(None), carrier, slice(None)), :]
            return data

    def extract_technology(self, data, tech_type, scenario):
        """Extracts the technology type specified by 'type'

        :param data: pd.DataFrame containing data to extract
        :param tech_type: technology type (e.g., conversion)
        :param scenario: scenario name
        :return: pd.DataFrame containing data of interest
        """
        # check if data contains technologies
        if "technology" not in data.index.dtypes:
            return data
        index_list = []
        # check if data contains technology and capacity_type index levels as it is the case for: capacity, ...
        if "technology" in data.index.dtypes and "capacity_type" in data.index.dtypes:
            if "location" in data.index.dtypes:
                #iterate over rows of data to find technologies with identical carrier
                for pos, index in enumerate(data.index):
                    if tech_type == "conversion":
                        if index[0] in self.results[scenario]["system"]["set_conversion_technologies"]:
                            index_list.append(index)
                    elif tech_type == "transport":
                        if index[0] in self.results[scenario]["system"]["set_transport_technologies"]:
                            index_list.append(index)
                    elif "storage" in tech_type:
                        if index[0] in self.results[scenario]["system"]["set_storage_technologies"]:
                            if "power" in tech_type and index[1] == "power":
                                index_list.append(index)
                            elif "energy" in tech_type and index[1] == "energy":
                                index_list.append(index)
                            elif "power" not in tech_type and "energy" not in tech_type:
                                index_list.append(index)
                    else:
                        warnings.warn(f"Technology type '{tech_type}' doesn't exist!")
        return data.loc[data.index.isin(index_list)]

    def extract_reference_carrier(self, data, carrier_type, scenario):
        """Extracts technologies of reference carrier type

        :param data: Data Frame containing set of technologies with different reference carriers
        :param carrier_type: String specifying reference carrier whose technologies should be extracted from data
        :param scenario: scenario of interest
        :return: Data Frame containing technologies of reference carrier only
        """
        reference_carriers = self.get_df("set_reference_carriers", is_set=True, scenario=scenario)
        if carrier_type not in [carrier for carrier in reference_carriers]:
            warnings.warn(f"Chosen reference carrier '{carrier_type}' doesn't exist")
            return data
        index_list = []
        for tech, carrier in enumerate(reference_carriers):
            if carrier == carrier_type:
                index_list.extend([index for index in data.index if index[0] == reference_carriers.index[tech]])

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
