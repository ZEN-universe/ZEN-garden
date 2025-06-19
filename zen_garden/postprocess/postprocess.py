"""
Class is defining the postprocessing of the results.
The class takes as inputs the optimization problem (model) and the system configurations (system).
The class contains methods to read the results and save them in a result dictionary (resultDict).
"""
import json
import logging
import os
from pathlib import Path

import numpy as np
import pint
from tables import NaturalNameWarning
import warnings
import pandas as pd
import xarray as xr
from filelock import FileLock
import yaml
from pydantic import BaseModel

from ..model.optimization_setup import OptimizationSetup


# Warnings
warnings.filterwarnings('ignore', category=NaturalNameWarning)

class Postprocess:
    """
    Class is defining the postprocessing of the results
    """
    def __init__(self, optimization_setup: OptimizationSetup, scenarios, model_name, subfolder=None, scenario_name=None, param_map=None):
        """postprocessing of the results of the optimization

        :param model: optimization model
        :param model_name: The name of the model used to name the output folder
        :param subfolder: The subfolder used for the results
        :param scenario_name: The name of the current scenario
        :param param_map: A dictionary mapping the parameters to the scenario names
        """
        logging.info("--- Postprocess results ---")
        # get the necessary stuff from the model
        self.optimization_setup = optimization_setup
        self.model = optimization_setup.model
        self.scenarios = scenarios
        self.system = optimization_setup.system
        self.analysis = optimization_setup.analysis
        self.solver = optimization_setup.solver
        self.energy_system = optimization_setup.energy_system
        self.params = optimization_setup.parameters
        self.vars = optimization_setup.variables
        self.sets = optimization_setup.sets
        self.constraints = optimization_setup.constraints
        self.param_map = param_map
        self.scaling = optimization_setup.scaling

        # get name or directory
        self.model_name = model_name
        self.name_dir = Path(self.analysis.folder_output).joinpath(self.model_name)

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
        self.overwrite = self.analysis.overwrite_output
        # get the compression param
        self.output_format = self.analysis.output_format

        # save everything
        self.save_sets()
        self.save_param()
        self.save_var()
        self.save_duals()
        self.save_system()
        self.save_analysis()
        self.save_scenarios()
        self.save_solver()
        self.save_unit_definitions()
        self.save_sequence_time_steps(scenario=scenario_name)
        self.save_param_map()
        if self.solver.run_diagnostics:
            self.save_benchmarking_data()

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
                self._write_h5_file(f_name, dictionary)

        elif format == "txt":
            f_name = f"{name}.txt"
            f_mode = "w+"

            # write if necessary
            if self.overwrite or not os.path.exists(f_name):
                with FileLock(f_name + ".lock").acquire(timeout=300):
                    with open(f_name, f_mode, encoding="utf-8") as outfile:
                        outfile.write(dictionary)
        else:
            raise AssertionError(f"The specified output format {format}, chosen in the config, is not supported")

    def save_benchmarking_data(self):
        """
        Saves the benchmarking data to a json file
        """
        # initialize dictionary
        benchmarking_data = dict()
        # get the benchmarking data
        benchmarking_data["objective_value"] = self.model.objective.value
        if self.solver.name == "gurobi":
            benchmarking_data["solving_time"] = self.model.solver_model.Runtime
            if "Method" in self.solver.solver_options:
                if self.solver.solver_options["Method"] == 2:
                    benchmarking_data["number_iterations"] = self.model.solver_model.BarIterCount
                else:
                    benchmarking_data["number_iterations"] = self.model.solver_model.IterCount
            benchmarking_data["solver_status"] = self.model.solver_model.Status
            benchmarking_data["number_constraints"] = self.model.solver_model.NumConstrs
            benchmarking_data["number_variables"] = self.model.solver_model.NumVars
        elif self.solver.name == "highs":
            benchmarking_data["solver_status"] = self.model.solver_model.getModelStatus().name
            benchmarking_data["solving_time"] = self.model.solver_model.getRunTime()
            benchmarking_data["number_iterations"] = self.model.solver_model.getInfo().simplex_iteration_count
            benchmarking_data["number_constraints"] = self.model.solver_model.getNumRow()
            benchmarking_data["number_variables"] = self.model.solver_model.getNumCol()
        else:
            logging.info(f"Saving benchmarking data for solver {self.solver.name} has not been implemented yet")

        benchmarking_data["scaling_time"] = self.scaling.scaling_time
        # get numerical range
        range_lhs, range_rhs = self.scaling.print_numerics(0, no_scaling=False,benchmarking_output= True)
        benchmarking_data["numerical_range_lhs"] = range_lhs
        benchmarking_data["numerical_range_rhs"] = range_rhs
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
        if not self.solver.save_parameters:
            logging.info("Parameters are not saved")
            return

        # dataframe serialization
        data_frames = {}
        for param in self.params.docs.keys():
            if self.solver.selected_saved_parameters and param not in self.solver.selected_saved_parameters:
                continue
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
            if self.solver.selected_saved_variables and name not in self.solver.selected_saved_variables:
                continue
            if name in self.vars.docs:
                doc = self.vars.docs[name]
                units = self.vars.units[name]
                index_list = self.get_index_list(doc)
            elif name.startswith("sos2_var"):
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
        if not self.solver.save_duals:
            logging.info("Duals are not saved")
            return

        # dataframe serialization
        data_frames = {}
        for name, arr in self.model.dual.items():
            if self.solver.selected_saved_duals and name not in self.solver.selected_saved_duals:
                continue
            if name in self.constraints.docs:
                doc = self.constraints.docs[name]
                index_list = self.get_index_list(doc)
            else:
                index_list = []
                doc = None
            # rescale
            if self.solver.use_scaling:
                cons_labels = self.model.constraints[name].labels.data
                scaling_factor = self.optimization_setup.scaling.D_r_inv[cons_labels]
                arr = arr * scaling_factor
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
        if hasattr(self.system,"fix_keys"):
            del self.system.fix_keys
        if hasattr(self.system,"i"):
            del self.system.i
        if self.system.use_rolling_horizon:
            fname = self.name_dir.parent.joinpath('system')
        else:
            fname = self.name_dir.joinpath('system')
        self.write_file(fname, self.system, format="json")

    def save_analysis(self):
        """
        Saves the analysis dict as json
        """
        if hasattr(self.analysis,"fix_keys"):
            del self.analysis.fix_keys
        if hasattr(self.analysis,"i"):
            del self.analysis.i
        if self.system.use_rolling_horizon:
            fname = self.name_dir.parent.joinpath('analysis')
        else:
            fname = self.name_dir.joinpath('analysis')
        # remove cwd path part to avoid saving the absolute path
        if os.path.isabs(self.analysis.dataset):
            self.analysis.dataset = os.path.split(Path(self.analysis.dataset))[-1]
            self.analysis.folder_output = os.path.split(Path(self.analysis.folder_output))[-1]
        self.write_file(fname, self.analysis, format="json")

    def save_solver(self):
        """
        Saves the solver dict as json
        """
        if hasattr(self.solver,"fix_keys"):
            del self.solver.fix_keys
        if hasattr(self.solver,"i"):
            del self.solver.i
        # This we only need to save once
        if self.system.use_rolling_horizon:
            fname = self.name_dir.parent.joinpath('solver')
        else:
            fname = self.name_dir.joinpath('solver')
        self.write_file(fname, self.solver, format="json")

    def save_scenarios(self):
        """
        Saves the scenario dict as json
        """
        # only save the scenarios at the highest level
        root_path = Path(self.analysis.folder_output).joinpath(self.model_name)
        fname = root_path.joinpath('scenarios')
        self.write_file(fname, self.scenarios, format="json")

    def save_unit_definitions(self):
        """
        Saves the user-defined units as txt
        """
        if self.system.use_rolling_horizon:
            fname = self.name_dir.parent.joinpath('unit_definitions')
        else:
            fname = self.name_dir.joinpath('unit_definitions')

        lines = []
        ureg = self.energy_system.unit_handling.ureg
        # Only save user-defined units (skip base units like 'meter')
        all_units = ureg._units
        default_units = pint.UnitRegistry()._units
        user_units = list(set(all_units.items()).difference(default_units.items()))
        for name, unit in user_units:
            if hasattr(unit, "raw") and f"{unit.raw}\n" not in lines:
                lines.append(f"{unit.raw}\n")
        txt = "".join(lines)
        self.write_file(fname, txt, format="txt")

    def save_param_map(self):
        """
        Saves the param_map dict as yaml
        """

        if self.param_map is not None:
            # This we only need to save once
            if self.system.use_rolling_horizon and self.system.conduct_scenario_analysis:
                fname = self.name_dir.parent.parent.joinpath('param_map')
            elif self.subfolder != Path(""):
                fname = self.name_dir.parent.joinpath('param_map')
            else:
                fname = self.name_dir.joinpath('param_map')
            self.write_file(fname, self.param_map, format="yml")

    def save_sequence_time_steps(self, scenario=None):
        """Saves the dict_all_sequence_time_steps dict as json

        :param scenario: name of scenario for which results are postprocessed
        """
        # extract and save sequence time steps, we transform the arrays to lists
        self.dict_sequence_time_steps = self.flatten_dict(self.energy_system.time_steps.get_sequence_time_steps_dict())
        self.dict_sequence_time_steps["optimized_time_steps"] = self.optimization_setup.optimized_time_steps
        self.dict_sequence_time_steps["time_steps_operation_duration"] = self.energy_system.time_steps.time_steps_operation_duration
        self.dict_sequence_time_steps["time_steps_storage_duration"] = self.energy_system.time_steps.time_steps_storage_duration
        self.dict_sequence_time_steps["time_steps_storage_level_startend_year"] = self.energy_system.time_steps.time_steps_storage_level_startend_year
        self.dict_sequence_time_steps["time_steps_year2operation"] = self.get_time_steps_year2operation()
        self.dict_sequence_time_steps["time_steps_year2storage"] = self.get_time_steps_year2storage()

        # add the scenario name
        if scenario is not None:
            add_on = f"_{scenario}"
        else:
            add_on = ""

            # This we only need to save once
        if self.system.use_rolling_horizon:
            fname = self.name_dir.parent.joinpath(f'dict_all_sequence_time_steps{add_on}')
        else:
            fname = self.name_dir.joinpath(f'dict_all_sequence_time_steps{add_on}')
        dict_sequence_time_steps = self.dict_sequence_time_steps
        dict_formatted = {}
        for k,v in dict_sequence_time_steps.items():
            if isinstance(v, np.ndarray):
                dict_formatted[k] = v.tolist()
            elif isinstance(v, dict):
                dict_formatted[k] = {str(kk): vv.tolist() if isinstance(vv, np.ndarray) else str(vv) for kk, vv in v.items()}
            elif isinstance(v, list):
                dict_formatted[k] = v
            else:
                NotImplementedError(f"Type {type(v)} not supported for key {k}")
        self.write_file(fname, dict_formatted, format="json")


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

        :param doc: docstring
        :return: index list
        """
        split_doc = doc.split(";")
        for string in split_doc:
            if "dims" in string:
                break
        string = string.replace("dims:", "")
        index_list = string.split(",")
        index_list_final = []
        for index in index_list:
            if index in self.analysis.header_data_inputs.keys():
                index_list_final.append(self.analysis.header_data_inputs[index])  # else:  #     pass  #     # index_list_final.append(index)
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

    def _transform_df(self, df, doc, units=None):
        """we transform the dataframe to a json string and load it into the dictionary as dict

        :param df: dataframe
        :param doc: doc string
        :param units: units
        :return: dictionary
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

    def _write_h5_file(self, file_name, dictionary,complevel=4,complib="blosc"):
        """Writes the dictionary to a hdf5 file

        :param file_name: The name of the file
        :param dictionary: The dictionary to save
        """
        if not self.overwrite and os.path.exists(file_name):
            raise FileExistsError("File already exists. Please set overwrite=True to overwrite the file.")
        with pd.HDFStore(file_name, mode='w', complevel=complevel, complib=complib) as store:
            for key, value in dictionary.items():
                if not isinstance(key, str):
                    raise TypeError("All dictionary keys must be strings!")
                if isinstance(value, dict):
                    input_dict, docstring, has_units = self._format_dict(value)
                    if not input_dict["dataframe"].empty:
                        store.put(key, input_dict["dataframe"], format='table')
                        # add additional attributes
                        index_names = input_dict["dataframe"].index.names
                        index_names = ",".join([str(name) for name in index_names])
                        store.get_storer(key).attrs.docstring = docstring
                        store.get_storer(key).attrs["name"] = key
                        store.get_storer(key).attrs["has_units"] = has_units
                        store.get_storer(key).attrs["index_names"] = index_names
                        # remove "_i_table" to reduce file size
                        try:
                            store.remove(key + "/_i_table")
                        except KeyError:
                            pass
                else:
                    raise TypeError(f"Type {type(value)} is not supported.")

    @staticmethod
    def _format_dict(input_dict):
        """ format the dictionary to be saved in the hdf file
        :param input_dict: The dictionary to format
        """
        expected_keys = ["dataframe", "docstring"]
        if "dataframe" in input_dict:
            df = input_dict["dataframe"]
            if not isinstance(df, pd.Series):
                if df.shape[1]:
                    df = df.squeeze(axis=1)
            input_dict["dataframe"] = df
        if "docstring" in input_dict:
            docstring = input_dict["docstring"]
        else:
            docstring = None
        if "units" in input_dict:
            units = input_dict["units"]
            assert isinstance(units, pd.Series), f"Units must be a pandas Series, but is {type(units)}"
            df = input_dict["dataframe"]
            assert units.index.intersection(df.index).equals(
                units.index), f"Units index {units.index} does not match dataframe index {df.index}"
            units.name = "units"
            df = pd.concat([df, units], axis=1)
            input_dict["dataframe"] = df
            has_units = True
        else:
            has_units = False
        if not (set(input_dict.keys()) == set(expected_keys) or set(input_dict.keys()) == set(expected_keys).union(
                ["units"])):
            raise ValueError(f"Expected keys are {expected_keys}, but got {input_dict.keys()}")
        return input_dict, docstring, has_units
