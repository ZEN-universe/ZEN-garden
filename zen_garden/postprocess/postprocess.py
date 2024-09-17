"""
:Title:        ZEN-GARDEN
:Created:      October-2021
:Authors:      Alissa Ganter (aganter@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Class is defining the postprocessing of the results.
The class takes as inputs the optimization problem (model) and the system configurations (system).
The class contains methods to read the results and save them in a result dictionary (resultDict).
"""
import json
import logging
import os
from pathlib import Path
import sys
import zlib
from tables import NaturalNameWarning
import warnings
import pandas as pd
import xarray as xr
from filelock import FileLock
import yaml
from pydantic import BaseModel

from ..utils import HDFPandasSerializer
from ..model.optimization_setup import OptimizationSetup


# Warnings
warnings.filterwarnings('ignore', category=NaturalNameWarning)

class Postprocess:
    """
    Class is defining the postprocessing of the results
    """
    def __init__(self, model: OptimizationSetup, scenarios, model_name, subfolder=None, scenario_name=None, param_map=None, include_year2operation=True):
        """postprocessing of the results of the optimization

        :param model: optimization model
        :param model_name: The name of the model used to name the output folder
        :param subfolder: The subfolder used for the results
        :param scenario_name: The name of the current scenario
        :param param_map: A dictionary mapping the parameters to the scenario names
        :param include_year2operation: Specify if the year2operation dict should be included in the results file
        """
        logging.info("--- Postprocess results ---")
        # get the necessary stuff from the model
        self.model = model.model
        self.scenarios = scenarios
        self.system = model.system
        self.analysis = model.analysis
        self.solver = model.solver
        self.energy_system = model.energy_system
        self.params = model.parameters
        self.vars = model.variables
        self.sets = model.sets
        self.constraints = model.constraints
        self.param_map = param_map
        self.scaling = model.scaling

        # get name or directory
        self.model_name = model_name
        self.name_dir = Path(self.analysis["folder_output"]).joinpath(self.model_name)

        # deal with the subfolder
        self.subfolder = subfolder
        # here we make use of the fact that None and "" both evaluate to False but any non-empty string doesn't
        if subfolder != Path(""):
            # check if mf within scenario analysis
            if isinstance(self.subfolder, tuple):
                scenario_dir = self.name_dir.joinpath(self.subfolder[0])
                os.makedirs(scenario_dir, exist_ok=True)
                mf_in_scenario_dir = self.subfolder[0].joinpath(self.subfolder[1])
                self.name_dir = self.name_dir.joinpath(mf_in_scenario_dir)
            else:
                self.name_dir = self.name_dir.joinpath(self.subfolder)
        # create the output directory
        os.makedirs(self.name_dir, exist_ok=True)

        # check if we should overwrite output
        self.overwrite = self.analysis["overwrite_output"]
        # get the compression param
        self.output_format = self.analysis["output_format"]

        # save everything
        self.save_sets()
        self.save_param()
        self.save_var()
        self.save_duals()
        self.save_system()
        self.save_analysis()
        self.save_scenarios()
        self.save_solver()
        self.save_param_map()
        if self.analysis.save_benchmarking_results:
            self.save_benchmarking_data()

        # extract and save sequence time steps, we transform the arrays to lists
        self.dict_sequence_time_steps = self.flatten_dict(self.energy_system.time_steps.get_sequence_time_steps_dict())
        self.dict_sequence_time_steps["optimized_time_steps"] = model.optimized_time_steps
        if include_year2operation:
            self.dict_sequence_time_steps["time_steps_year2operation"] = self.get_time_steps_year2operation()
            self.dict_sequence_time_steps["time_steps_year2storage"] = self.get_time_steps_year2storage()

        self.save_sequence_time_steps(scenario=scenario_name)

    def write_file(self, name, dictionary, format=None):
        """Writes the dictionary to file as json, if compression attribute is True, the serialized json is compressed
            and saved as binary file

        :param name: Filename without extension
        :param dictionary: The dictionary to save
        :param format: Force the format to use, if None use output_format attribute of instance
        """

        if isinstance(dictionary, BaseModel):
            dictionary = dictionary.model_dump()

        # set the format
        if format is None:
            format = self.output_format

        if format == "yml":
            # serialize to string
            serialized_dict = yaml.dump(dictionary)

            # prep output file
            f_name = f"{name}.yml"
            f_mode = "w"

            # write if necessary
            if self.overwrite or not os.path.exists(f_name):
                with FileLock(f_name + ".lock").acquire(timeout=300):
                    with open(f_name, f_mode) as outfile:
                        outfile.write(serialized_dict)

        elif format == "json":
            # serialize to string

            serialized_dict = json.dumps(dictionary, indent=2)

            # write normal json
            f_name = f"{name}.json"
            f_mode = "w+"

            # write if necessary
            if self.overwrite or not os.path.exists(f_name):
                with FileLock(f_name + ".lock").acquire(timeout=300):
                    with open(f_name, f_mode) as outfile:
                        outfile.write(serialized_dict)

        elif format == "h5":
            f_name = f"{name}.h5"
            with FileLock(f_name + ".lock").acquire(timeout=300):
                HDFPandasSerializer.serialize_dict(file_name=f_name, dictionary=dictionary, overwrite=self.overwrite)

        else:
            raise AssertionError(f"The specified output format {format}, chosen in the config, is not supported")

    def save_benchmarking_data(self):
        #initialize dictionary
        benchmarking_data = {}
        # get the benchmarking data
        benchmarking_data["solving_time"] = self.model.solver_model.Runtime
        if self.solver.solver_options["Method"] == 2:
            benchmarking_data["number_iterations"] = self.model.solver_model.BarIterCount
        else:
            benchmarking_data["number_iterations"] = self.model.solver_model.IterCount
        benchmarking_data["solver_status"] = self.model.solver_model.Status
        benchmarking_data["objective_value"] = self.model.objective_value
        benchmarking_data["scaling_time"] = self.scaling.scaling_time

        #get numerical range
        range_lhs, range_rhs, cond = self.scaling.print_numerics(0, False, True)
        benchmarking_data["numerical_range_lhs"] = range_lhs
        benchmarking_data["numerical_range_rhs"] = range_rhs
        benchmarking_data["condition_number"] = cond


        fname = self.name_dir.joinpath('benchmarking')
        self.write_file(fname, benchmarking_data, format="json")

    def save_sets(self):
        """ Saves the Set values to a json file which can then be
        post-processed immediately or loaded and postprocessed at some other time"""
        # dataframe serialization
        data_frames = {}
        for set in self.sets:
            if not set.is_indexed():
                continue
            vals = set.data
            index_name = [set.name]

            # if the returned dict is emtpy we create a nan value
            if len(vals) == 0:
                indices = pd.Index(data=[], name=index_name[0])
                data = []
            else:
                indices = list(vals.keys())
                data = list(vals.values())
                data_strings = []
                for tpl in data:
                    string = ""
                    for ind, t in enumerate(tpl):
                        if ind == len(tpl) - 1:
                            string += str(t)
                        else:
                            string += str(t) + ","
                    data_strings.append(string)
                data = data_strings

                # create a multi index if necessary
                if len(indices) >= 1 and isinstance(indices[0], tuple):
                    if len(index_name) == len(indices[0]):
                        indices = pd.MultiIndex.from_tuples(indices, names=index_name)
                    else:
                        indices = pd.MultiIndex.from_tuples(indices)
                else:
                    if len(index_name) == 1:
                        indices = pd.Index(data=indices, name=index_name[0])
                    else:
                        indices = pd.Index(data=indices)

            # create dataframe
            df = pd.DataFrame(data=data, columns=["value"], index=indices)
            # update dict
            doc = self.sets.docs[set.name]
            data_frames[index_name[0]] = self._transform_df(df,doc)

        self.write_file(self.name_dir.joinpath('set_dict'), data_frames)

    def save_param(self):
        """ Saves the Param values to a json file which can then be
        post-processed immediately or loaded and postprocessed at some other time"""

        # dataframe serialization
        data_frames = {}
        for param in self.params.docs.keys():
            # get the values
            vals = getattr(self.params, param)
            doc = self.params.docs[param]
            units = self.params.units[param]
            index_list = self.get_index_list(doc)
            # data frame
            if isinstance(vals, xr.DataArray):
                df = vals.to_dataframe("value").dropna()
            # we have a scalar
            else:
                df = pd.DataFrame(data=[vals], columns=["value"])

            # rename the index
            if len(df.index.names) == len(index_list):
                df.index.names = index_list

            units = self._unit_df(units,df.index)
            # update dict
            data_frames[param] = self._transform_df(df, doc, units)

        # write to json
        self.write_file(self.name_dir.joinpath('param_dict'), data_frames)

    def save_var(self):
        """ Saves the variable values to a json file which can then be
        post-processed immediately or loaded and postprocessed at some other time"""

        # dataframe serialization
        data_frames = {}
        for name, arr in self.model.solution.items():
            if name in self.vars.docs:
                doc = self.vars.docs[name]
                units = self.vars.units[name]
                index_list = self.get_index_list(doc)
            elif name.startswith("sos2_var") or name in ["tech_on_var", "tech_off_var"]:
                continue
            else:
                index_list = []
                doc = None
                units = None

            # create dataframe
            df = arr.to_dataframe("value").dropna()
            # rename the index
            if len(df.index.names) == len(index_list):
                df.index.names = index_list

            units = self._unit_df(units,df.index)
            # we transform the dataframe to a json string and load it into the dictionary as dict
            data_frames[name] = self._transform_df(df,doc,units)

        self.write_file(self.name_dir.joinpath('var_dict'), data_frames)

    def save_duals(self):
        """ Saves the dual variable values to a json file which can then be
        post-processed immediately or loaded and postprocessed at some other time"""
        if not self.solver["add_duals"]:
            return

        # dataframe serialization
        data_frames = {}
        for name, arr in self.model.dual.items():
            if name in self.constraints.docs:
                doc = self.constraints.docs[name]
                index_list = self.get_index_list(doc)
            else:
                index_list = []
                doc = None

            # create dataframe
            if len(arr.shape) > 0:
                df = arr.to_series().dropna()
            else:
                df = pd.DataFrame(data=[arr.values], columns=["value"])

            # rename the index
            if len(df.index.names) == len(index_list):
                df.index.names = index_list

            # we transform the dataframe to a json string and load it into the dictionary as dict
            data_frames[name] = self._transform_df(df,doc)

        self.write_file(self.name_dir.joinpath('dual_dict'), data_frames)

    def save_system(self):
        """
        Saves the system dict as json
        """
        if self.system["use_rolling_horizon"]:
            fname = self.name_dir.parent.joinpath('system')
        else:
            fname = self.name_dir.joinpath('system')
        self.write_file(fname, self.system, format="json")

    def save_analysis(self):
        """
        Saves the analysis dict as json
        """
        if self.system["use_rolling_horizon"]:
            fname = self.name_dir.parent.joinpath('analysis')
        else:
            fname = self.name_dir.joinpath('analysis')
        self.write_file(fname, self.analysis, format="json")

    def save_scenarios(self):
        """
        Saves the analysis dict as json
        """

        # This we only need to save once
        # check if MF within scenario analysis
        if isinstance(self.subfolder, tuple):
            # check if there are sub_scenarios (parent must then be the name of the parent scenario)
            if not self.subfolder[0].parent == Path("."):
                fname = self.name_dir.parent.parent.parent.joinpath('scenarios')
            else:
                # MF with in scenario analysis without sub-scenarios
                fname = self.name_dir.parent.parent.joinpath('scenarios')
        # only MF or only scenario analysis
        elif self.subfolder != Path(""):
            fname = self.name_dir.parent.joinpath('scenarios')
        # neither MF nor scenario analysis
        else:
            fname = self.name_dir.joinpath('scenarios')
        self.write_file(fname, self.scenarios, format="json")

    def save_solver(self):
        """
        Saves the solver dict as json
        """

        # This we only need to save once
        if self.system["use_rolling_horizon"]:
            fname = self.name_dir.parent.joinpath('solver')
        else:
            fname = self.name_dir.joinpath('solver')
        self.write_file(fname, self.solver, format="json")

    def save_param_map(self):
        """
        Saves the param_map dict as yaml
        """

        if self.param_map is not None:
            # This we only need to save once
            if self.system["use_rolling_horizon"] and self.system["conduct_scenario_analysis"]:
                fname = self.name_dir.parent.parent.joinpath('param_map')
            elif self.subfolder != Path(""):
                fname = self.name_dir.parent.joinpath('param_map')
            else:
                fname = self.name_dir.joinpath('param_map')
            self.write_file(fname, self.param_map, format="yml")

    def save_sequence_time_steps(self, scenario=None):
        """Saves the dict_all_sequence_time_steps dict as json

        :param scenario: #TODO describe parameter/return
        """
        # add the scenario name
        if scenario is not None:
            add_on = f"_{scenario}"
        else:
            add_on = ""

            # This we only need to save once
        if self.system["use_rolling_horizon"]:
            fname = self.name_dir.parent.joinpath(f'dict_all_sequence_time_steps{add_on}')
        else:
            fname = self.name_dir.joinpath(f'dict_all_sequence_time_steps{add_on}')
        self.write_file(fname, self.dict_sequence_time_steps)

    def _transform_df(self, df, doc, units=None):
        """we transform the dataframe to a json string and load it into the dictionary as dict

        :param df: #TODO describe parameter/return
        :param doc: #TODO describe parameter/return
        :return: #TODO describe parameter/return
        """
        if self.output_format == "h5":
            if units is not None:
                dataframe = {"dataframe": df, "docstring": doc, "units": units}
            else:
                dataframe = {"dataframe": df, "docstring": doc}
        else:
            raise AssertionError(f"The specified output format {self.output_format}, chosen in the config, is not supported")
        return dataframe

    def _doc_to_df(self, doc):
        """Transforms the docstring to a dataframe

        :param doc: doc string
        :return: pd.Series of the docstring
        """
        if doc is not None:
            return pd.Series(doc.split(";")).str.split(":",expand=True).set_index(0).squeeze()
        else:
            return pd.DataFrame()

    def _unit_df(self, units, index):
        """Transforms the units to a series

        :param units: units string
        :param index: index of the target dataframe
        :return: pd.Series of the units
        """
        if units is not None:
            if isinstance(units, str):
                return pd.Series(units, index=index)
            elif len(units) == len(index):
                units.index.names = index.names
                return units
            else:
                raise AssertionError("The length of the units does not match the length of the index")
        else:
            return None

    def flatten_dict(self, dictionary):
        """Creates a copy of the dictionary where all numpy arrays are recursively flattened to lists such that it can
            be saved as json file

        :param dictionary: The input dictionary
        :return: A copy of the dictionary containing lists instead of arrays
        """
        # create a copy of the dict to avoid overwrite
        out_dict = dict()

        # falten all arrays
        for k, v in dictionary.items():
            # transform the key None to 'null'
            if k is None:
                k = 'null'

            # recursive call
            if isinstance(v, dict):
                out_dict[k] = self.flatten_dict(v)  # flatten the array to list
            elif isinstance(v, pd.Series):
                # Note: list(v) creates a list of np objects v.tolist() not
                out_dict[k] = v.values.tolist()
            # take as is
            else:
                out_dict[k] = v

        return out_dict

    def get_index_list(self, doc):
        """ get index list from docstring

        :param doc: #TODO describe parameter/return
        :return: #TODO describe parameter/return
        """
        split_doc = doc.split(";")
        for string in split_doc:
            if "dims" in string:
                break
        string = string.replace("dims:", "")
        index_list = string.split(",")
        index_list_final = []
        for index in index_list:
            if index in self.analysis["header_data_inputs"].keys():
                index_list_final.append(self.analysis["header_data_inputs"][index])  # else:  #     pass  #     # index_list_final.append(index)
        return index_list_final

    def get_time_steps_year2operation(self):
        """ Returns a HDF5-Serializable version of the dict_time_steps_year2operation dictionary."""
        ans = {}
        for year, time_steps in self.energy_system.time_steps.time_steps_year2operation.items():
            ans[str(year)] = time_steps
        return ans

    def get_time_steps_year2storage(self):
        """ Returns a HDF5-Serializable version of the dict_time_steps_year2storage dictionary."""
        ans = {}
        for year, time_steps in self.energy_system.time_steps.time_steps_year2storage.items():
            ans[str(year)] = time_steps
        return ans
