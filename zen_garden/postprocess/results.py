"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Class is defining the postprocessing of the results.
              The class takes as inputs the optimization problem (model) and the system configurations (system).
              The class contains methods to read the results and save them in a result dictionary (resultDict).
==========================================================================================================================================================================="""
import sys

import numpy as np
import pyomo.environ as pe
import pandas as pd
import pathlib
import shutil
import json
import zlib
import os

from ..model.objects.energy_system import EnergySystem
from ..model.objects.component import Parameter,Variable,Constraint
from ..utils import RedirectStdStreams

class Postprocess:

    def __init__(self, model, scenarios, model_name, subfolder=None):
        """postprocessing of the results of the optimization
        :param model: optimization model
        :param model_name: The name of the model used to name the output folder
        :param subfolder: The subfolder used for the results
        """

        # get the necessary stuff from the model
        self.model = model.model
        self.scenarios = scenarios
        self.system = model.system
        self.analysis = model.analysis
        self.solver = model.solver
        self.opt = model.opt
        self.params         = Parameter.get_component_object()
        self.vars           = Variable.get_component_object()
        self.constraints    = Constraint.get_component_object()

        # get name or directory
        self.model_name = model_name
        self.nameDir = pathlib.Path(self.analysis["folder_output"]).joinpath(self.model_name)

        # deal with the subfolder
        self.subfolder = subfolder
        # here we make use of the fact that None and "" both evaluate to False but any non-empty string doesn't
        if self.subfolder:
            self.nameDir = self.nameDir.joinpath(self.subfolder)
        # create the output directory
        os.makedirs(self.nameDir, exist_ok=True)

        # get the compression param
        self.compress = self.analysis["compress_output"]

        # save the pyomo yml
        if self.analysis["write_results_yml"]:
            with RedirectStdStreams(open(os.path.join(self.nameDir, "results.yml"), "w+")):
                model.results.write()

        # save everything
        self.saveParam()
        self.saveVar()
        self.saveSystem()
        self.saveAnalysis()
        self.saveScenarios()
        self.saveSolver()
        self.saveOpt()

        # extract and save sequence time steps, we transform the arrays to lists
        self.dictSequenceTimeSteps = self.flatten_dict(EnergySystem.get_sequence_time_steps_dict())
        self.saveSequenceTimeSteps()

        # case where we should run the post-process as normal
        if model.analysis['postprocess']:
            pass
            # TODO: implement this...
            #self.process()

    def write_file(self, name, dictionary):
        """
        Writes the dictionary to file as json, if compression attribute is True, the serialized json is compressed
        and saved as binary file
        :param name: Filename without extension
        :param dictionary: The dictionary to save
        """

        # serialize to string
        serialized_dict = json.dumps(dictionary, indent=2)

        # if the string is larger than the max output size we compress anyway
        force_compression = False
        if not self.compress and sys.getsizeof(serialized_dict)/1024**2 > self.analysis["max_output_size_mb"]:
            print(f"WARNING: The file {name}.json would be larger than the maximum allowed output size of "
                  f"{self.analysis['max_output_size_mb']}MB, compressing...")
            force_compression = True

        if self.compress or force_compression:
            # compress and write
            compressed = zlib.compress(serialized_dict.encode())
            with open(f"{name}.gzip", "wb") as outfile:
                outfile.write(compressed)
        else:
            # write normal json
            with open(f"{name}.json", "w+") as outfile:
                outfile.write(serialized_dict)

    def saveParam(self):
        """ Saves the Param values to a json file which can then be
        post-processed immediately or loaded and postprocessed at some other time"""

        # dataframe serialization
        data_frames = {}
        for param in self.params.docs.keys():
            # get the values
            vals = getattr(self.params, param)
            doc = self.params.docs[param]
            index_list = self.getIndexList(doc)
            if len(index_list) == 0:
                index_names = None
            elif len(index_list) == 1:
                index_names = index_list[0]
            else:
                index_names = index_list
            # create a dictionary if necessary
            if not isinstance(vals, dict):
                indices = pd.Index(data=[0],name=index_names)
                data = [vals]
            # if the returned dict is emtpy we create a nan value
            elif len(vals) == 0:
                if len(index_list)>1:
                    indices = pd.MultiIndex(levels=[[]]*len(index_names),codes=[[]]*len(index_names),names=index_names)
                else:
                    indices = pd.Index(data=[],name=index_names)
                data = []
            # we read out everything
            else:
                indices = list(vals.keys())
                data = list(vals.values())

                # create a multi index if necessary
                if len(indices)>=1 and isinstance(indices[0],tuple):
                    if len(index_list) == len(indices[0]):
                        indices = pd.MultiIndex.from_tuples(indices,names=index_names)
                    else:
                        indices = pd.MultiIndex.from_tuples(indices)
                else:
                    if len(index_list) == 1:
                        indices = pd.Index(data=indices,name=index_names)
                    else:
                        indices = pd.Index(data=indices)

            # create dataframe
            df = pd.DataFrame(data=data, columns=["value"], index=indices)

            # update dict
            data_frames[param] = {"dataframe": json.loads(df.to_json(orient="table", indent=2)),
                                  "docstring": doc}

        # write to json
        self.write_file(self.nameDir.joinpath('paramDict'), data_frames)

    def saveVar(self):
        """ Saves the variable values to a json file which can then be
        post-processed immediately or loaded and postprocessed at some other time"""

        # dataframe serialization
        data_frames = {}
        for var in self.model.component_objects(pe.Var, active=True):
            if var.name in self.vars.docs:
                doc = self.vars.docs[var.name]
                index_list = self.getIndexList(doc)
                if len(index_list) == 0:
                    index_names = None
                elif len(index_list) == 1:
                    index_names = index_list[0]
                else:
                    index_names = index_list
            else:
                index_list = []
                doc = None
            # get indices and values
            indices = [index for index in var]
            values = [getattr(var[index], "value", None) for index in indices]

            # create a multi index if necessary
            if len(indices)>=1 and isinstance(indices[0], tuple):
                if len(index_list) == len(indices[0]):
                    indices = pd.MultiIndex.from_tuples(indices, names=index_names)
                else:
                    indices = pd.MultiIndex.from_tuples(indices)
            else:
                if len(index_list) == 1:
                    indices = pd.Index(data=indices, name=index_names)
                else:
                    indices = pd.Index(data=indices)

            # create dataframe
            df = pd.DataFrame(data=values, columns=["value"], index=indices)

            # we transform the dataframe to a json string and load it into the dictionary as dict
            data_frames[var.name] = {"dataframe": json.loads(df.to_json(orient="table", indent=2)),
                                     "docstring": doc}

        self.write_file(self.nameDir.joinpath('varDict'), data_frames)

    def saveSystem(self):
        """
        Saves the system dict as json
        """

        # This we only need to save once
        if self.subfolder:
            fname = self.nameDir.parent.joinpath('System')
        else:
            fname = self.nameDir.joinpath('System')
        if not fname.exists():
            self.write_file(fname, self.system)

    def saveAnalysis(self):
        """
        Saves the analysis dict as json
        """

        # This we only need to save once
        if self.subfolder:
            fname = self.nameDir.parent.joinpath('Analysis')
        else:
            fname = self.nameDir.joinpath('Analysis')
        if not fname.exists():
            self.write_file(fname, self.analysis)

    def saveScenarios(self):
        """
        Saves the analysis dict as json
        """

        # This we only need to save once
        if self.subfolder:
            fname = self.nameDir.parent.joinpath('Scenarios')
        else:
            fname = self.nameDir.joinpath('Scenarios')
        if not fname.exists():
            self.write_file(fname, self.scenarios)

    def saveSolver(self):
        """
        Saves the solver dict as json
        """

        # This we only need to save once
        if self.subfolder:
            fname = self.nameDir.parent.joinpath('Solver')
        else:
            fname = self.nameDir.joinpath('Solver')
        if not fname.exists():
            self.write_file(fname, self.solver)

    def saveOpt(self):
        """
        Saves the opt dict as json
        """
        if self.solver["name"] != "gurobi_persistent":
            self.write_file(self.nameDir.joinpath('optDict'), self.opt.__dict__)

            # copy the log file
            shutil.copy2(os.path.abspath(self.opt._log_file), self.nameDir)

    def saveSequenceTimeSteps(self):
        """
        Saves the dict_all_sequence_time_steps dict as json
        """

        # This we only need to save once
        if self.subfolder:
            fname = self.nameDir.parent.joinpath('dict_all_sequence_time_steps')
        else:
            fname = self.nameDir.joinpath('dict_all_sequence_time_steps')
        if not fname.exists():
            self.write_file(fname, self.dictSequenceTimeSteps)

    def flatten_dict(self, dictionary):
        """
        Creates a copy of the dictionary where all numpy arrays are recursively flattened to lists such that it can
        be saved as json file
        :param dictionary: The input dictionary
        :return: A copy of the dictionary containing lists instead of arrays
        """
        # create a copy of the dict to avoid overwrite
        dictionary = dictionary.copy()

        # faltten all arrays
        for k, v in dictionary.items():
            # recursive call
            if isinstance(v, dict):
                dictionary[k] = self.flatten_dict(v)
                # flatten the array to list
            elif isinstance(v, np.ndarray):
                # Note: list(v) creates a list of np objects v.tolist() not
                dictionary[k] = v.tolist()
            # take as is
            else:
                dictionary[k] = v

        return dictionary

    def getIndexList(self,doc):
        """ get index list from docstring """
        splitDoc = doc.split(";")
        for string in splitDoc:
            if "dims" in string:
                break
        string = string.replace("dims:","")
        index_list = string.split(",")
        indexListFinal = []
        for index in index_list:
            if index in self.analysis["header_data_inputs"].keys():
                indexListFinal.append(self.analysis["header_data_inputs"][index])
            else:
                pass
                # indexListFinal.append(index)
        return indexListFinal

class Results(object):
    """
    This class reads in the results after the pipeline has run
    """

    def __init__(self, path, load_opt=False):
        """
        Initializes the Results class with a given path
        :param path: Path to the output of the optimization problem
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
        self.results["dictSequenceTimeSteps"] = self.load_sequence_time_steps(self.path)

        # get the years
        self.years = list(range(0, self.results["system"]["optimized_years"]))

        # check what type of results we have
        if self.results["system"]["conduct_scenario_analysis"]:
            self.has_scenarios = True
            self.scenarios = [f"scenario_{scenario}" for scenario in self.results["scenarios"].keys()]
        else:
            self.has_scenarios = False
            self.scenarios = [None]
        if self.results["system"]["useRollingHorizon"]:
            self.has_MF = True
            self.mf = [f"MF_{step_horizon}" for step_horizon in self.years]
        else:
            self.has_MF = False
            self.mf = [None]

        # cycle through the dirs
        for scenario in self.scenarios:
            # init dict
            self.results[scenario] = {}
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
                self.results[scenario][mf]["pars_and_vars"].update(self.load_params(current_path))
                self.results[scenario][mf]["pars_and_vars"].update(self.load_vars(current_path))

                # the opt we only load when requested
                if load_opt:
                    self.results[scenario][mf]["optdict"] = self.load_opt(current_path)

        # load the time step duration
        self.timeStepOperationalDuration = self.loadTimeStepOperationDuration()
        self.timeStepStorageDuration = self.loadTimeStepStorageDuration()

    @classmethod
    def _read_file(cls, name):
        """
        Reads out a file and decompresses it if necessary
        :param name: File name without extension
        :return: The decompressed content of the file as string
        """

        # compressed version
        if os.path.exists(f"{name}.gzip"):
            with open(f"{name}.gzip", "rb") as f:
                content_compressed = f.read()
            return zlib.decompress(content_compressed).decode()

        # normal version
        if os.path.exists(f"{name}.json"):
            with open(f"{name}.json", "r") as f:
                content = f.read()
            return content

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

            # the dataframe we transform to an actual dataframe
            dict_df[key]['dataframe'] = pd.read_json(json.dumps(dict_raw[key]['dataframe']),
                                                     orient="table")

        return dict_df

    @classmethod
    def load_params(cls, path):
        """
        Loads the parameter dict from a given path
        :param path: Path to load the parameter dict from
        :return: The parameter dict
        """

        # load the raw dict
        raw_dict = cls._read_file(os.path.join(path, "paramDict"))
        paramDict_raw = json.loads(raw_dict)

        return cls._dict2df(paramDict_raw)

    @classmethod
    def load_vars(cls, path):
        """
        Loads the var dict from a given path
        :param path: Path to load the var dict from
        :return: The var dict
        """

        # load the raw dict
        raw_dict = cls._read_file(os.path.join(path, "varDict"))
        varDict_raw = json.loads(raw_dict)

        return cls._dict2df(varDict_raw)

    @classmethod
    def load_system(cls, path):
        """
        Loads the system dict from a given path
        :param path: Directory to load the dictionary from
        :return: The system dictionary
        """

        # get the dict
        raw_dict = cls._read_file(os.path.join(path, "System"))
        system_dict = json.loads(raw_dict)

        return system_dict

    @classmethod
    def load_analysis(cls, path):
        """
        Loads the analysis dict from a given path
        :param path: Directory to load the dictionary from
        :return: The analysis dictionary
        """

        # get the dict
        raw_dict = cls._read_file(os.path.join(path, "Analysis"))
        analysis_dict = json.loads(raw_dict)

        return analysis_dict

    @classmethod
    def load_solver(cls, path):
        """
        Loads the solver dict from a given path
        :param path: Directory to load the dictionary from
        :return: The analysis dictionary
        """

        # get the dict
        raw_dict = cls._read_file(os.path.join(path, "Solver"))
        solver_dict = json.loads(raw_dict)

        return solver_dict

    @classmethod
    def load_scenarios(cls, path):
        """
        Loads the scenarios dict from a given path
        :param path: Directory to load the dictionary from
        :return: The analysis dictionary
        """

        # get the dict
        raw_dict = cls._read_file(os.path.join(path, "Scenarios"))
        scenarios_dict = json.loads(raw_dict)

        return scenarios_dict

    @classmethod
    def load_opt(cls, path):
        """
        Loads the opt dict from a given path
        :param path: Directory to load the dictionary from
        :return: The analysis dictionary
        """

        # get the dict
        raw_dict = cls._read_file(os.path.join(path, "optDict"))
        opt_dict = json.loads(raw_dict)

        return opt_dict

    @classmethod
    def load_sequence_time_steps(cls, path):
        """
        Loads the dictSequenceTimeSteps from a given path
        :param path: Path to load the dict from
        :return: dictSequenceTimeSteps
        """

        # get the dict
        raw_dict = cls._read_file(os.path.join(path, "dict_all_sequence_time_steps"))
        dictSequenceTimeSteps = json.loads(raw_dict)

        # json string None to 'null'
        dictSequenceTimeSteps['yearly'][None] = dictSequenceTimeSteps['yearly']['null']
        del dictSequenceTimeSteps['yearly']['null']

        # tranform all lists to arrays
        return cls.expand_dict(dictSequenceTimeSteps)

    @classmethod
    def expand_dict(cls, dictionary):
        """
        Creates a copy of the dictionary where all lists are recursively transformed to numpy arrays
        :param dictionary: The input dictionary
        :return: A copy of the dictionary containing arrays instead of lists
        """
        # create a copy of the dict to avoid overwrite
        dictionary = dictionary.copy()

        # faltten all arrays
        for k, v in dictionary.items():
            # recursive call
            if isinstance(v, dict):
                dictionary[k] = cls.expand_dict(v)
                # flatten the array to list
            elif isinstance(v, list):
                # Note: list(v) creates a list of np objects v.tolist() not
                dictionary[k] = np.array(v)
            # take as is
            else:
                dictionary[k] = v

        return dictionary

    def get_df(self, name, isStorage=False, scenario=None, to_csv=None, csv_kwargs=None):
        """
        Extracts the dataframe from the results
        :param name: The name of the dataframe to extract
        :param isStorage: Whether it is a storage or not
        :param scenario: If multiple scenarios are in the results, only consider this one
        :param to_csv: Save the dataframes to a csv file
        :param csv_kwargs: additional keyword arguments forwarded to the to_csv method of pandas
        :return: The dataframe that should have been extracted. If multiple scenarios are present a dictionary
                 with scenarios as keys and dataframes as value is returned
        """

        # set the dict
        EnergySystem.set_sequence_time_steps_dict(self.results["dictSequenceTimeSteps"])

        # select the scenarios
        if scenario is not None:
            scenarios = [scenario]
        else:
            scenarios = self.scenarios

        # loop
        _data = {}
        for scenario in scenarios:
            if not self.has_MF:
                # we set the dataframe of the variable into the data dict
                _data[scenario] = self.results[scenario][None]["pars_and_vars"][name]["dataframe"]

            else:
                # init the scenario
                _mf_data = {}

                # cycle through all MFs
                for year, mf in enumerate(self.mf):
                    _var = self.results[scenario][mf]["pars_and_vars"][name]["dataframe"]

                    # single element that is not a year
                    if len(_var) == 1 and not np.isfinite(_var.index[0]):
                        _data[scenario] = _var
                        break
                    # if the year is in the index (no multiindex)
                    elif year in _var.index:
                        _mf_data[year] = _var.loc[year]
                        yearlyComponent = True
                    else:
                        # unstack the year
                        _varSeries = _var.unstack()
                        # if all columns in years (drop the value level)
                        if _varSeries.columns.droplevel(0).difference(self.years).empty:
                            # get the data
                            tmp_data = _varSeries[("value", year)]
                            # rename
                            tmp_data.name = "value"
                            # set
                            _mf_data[year] = tmp_data
                            yearlyComponent = True
                        # if more time steps than years, then it is operational ts (we drop value in columns)
                        elif pd.to_numeric(_varSeries.columns.droplevel(0),
                                           errors="coerce").equals(_varSeries.columns.droplevel(0)):
                            # TODO only valid for same time steps between techs
                            if isStorage:
                                techProxy = [k for k in self.results["dictSequenceTimeSteps"]["operation"].keys()
                                             if "storage" in k.lower()][0]
                            else:
                                techProxy = [k for k in self.results["dictSequenceTimeSteps"]["operation"].keys()
                                             if "storage" not in k.lower()][0]
                            # get the timesteps
                            timeStepsYear = EnergySystem.encode_time_step(techProxy,
                                                                        EnergySystem.decode_time_step(None, year,
                                                                                                    "yearly"),
                                                                        yearly=True)
                            # get the data
                            tmp_data = _varSeries[[("value", tstep) for tstep in timeStepsYear]]
                            # rename
                            tmp_data.name = "value"
                            # set
                            _mf_data[year] = tmp_data
                            yearlyComponent = False
                        # else not a time index
                        else:
                            _data[scenario] = _varSeries.stack()
                            break
                # This is a for-else, it is triggered if we did not break the loop
                else:
                    # deal with the years
                    if yearlyComponent:
                        # concat
                        _df = pd.concat(_mf_data, axis=0, keys=_mf_data.keys())
                        _dfIndex = _df.index.copy()
                        for level, codes in enumerate(_df.index.codes):
                            if len(np.unique(codes)) == 1 and np.unique(codes) == 0:
                                _dfIndex = _dfIndex.droplevel(level)
                                break
                        _df.index = _dfIndex
                        if year not in _var.index:
                            _indexSort = list(range(0, _df.index.nlevels))
                            _indexSort.append(_indexSort[0])
                            _indexSort.pop(0)
                            _df = _df.reorder_levels(_indexSort)
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

    def loadTimeStepOperationDuration(self):
        """
        Loads duration of operational time steps
        """
        return self.get_df("time_steps_operation_duration")

    def loadTimeStepStorageDuration(self):
        """
        Loads duration of operational time steps
        """
        return self.get_df("time_steps_storage_level_duration")

    def getFullTS(self, component, element_name=None, year=None, scenario=None):
        """
        Calculates the full timeseries for a given element
        :param component: Either the dataframe of a component as pandas.Series or the name of the component
        :param element_name: The name of the element
        :param scenario: The scenario for with the component should be extracted (only if needed)
        :return: A dataframe containing the full timeseries of the element
        """
        # extract the data
        component_name, component_data = self._get_component_data(component, scenario)

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
            _storageString = ""
        else:
            _storageString = "_storage_level"

        # calculate the full time series
        _outputTemp = {}
        for row in component_data.index:
            # we know the name
            if element_name:
                _sequence_time_steps = EnergySystem.get_sequence_time_steps(element_name+_storageString)
            # we extract the name
            else:
                _sequence_time_steps = EnergySystem.get_sequence_time_steps(row[0]+_storageString)

            # throw together
            _sequence_time_steps = _sequence_time_steps[np.in1d(_sequence_time_steps,list(component_data.columns))]
            _outputTemp[row] = component_data.loc[row,_sequence_time_steps].reset_index(drop=True)
            if year is not None:
                if year in self.years:
                    hours_of_year = self._get_hours_of_year(year)
                    _outputTemp[row] = (_outputTemp[row][hours_of_year]).reset_index(drop=True)
                else:
                    print(f"WARNING: year {year} not in years {self.years}. Return component values for all years")

        # concat and return
        outputDf = pd.concat(_outputTemp,axis=0,keys = _outputTemp.keys()).unstack()
        return outputDf

    def getTotal(self, component, element_name=None, year=None, scenario=None,split_years = True):
        """
        Calculates the total Value of a component
        :param component: Either a dataframe as returned from <get_df> or the name of the component
        :param element_name: The element name to calculate the value for, defaults to all elements
        :param year: The year to calculate the value for, defaults to all years
        :param scenario: The scenario to calculate the total value for
        :return: A dataframe containing the total value with the specified paramters
        """
        # extract the data
        component_name,component_data = self._get_component_data(component,scenario)

        ts_type = self._get_ts_type(component_data,component_name)

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
            _storageString = ""
        else:
            _isStorage = True
            _storageString = "_storage_level"

        # extract time step duration
        timeStepDuration = self._get_ts_duration(scenario,is_storage=_isStorage)

        # If we have an element name
        if elementName is not None:
            # check that it is in the index
            assert elementName in component_data.index.get_level_values(level=0), \
                f"element {elementName} is not found in index of {component_name}"
            # get the index
            component_data = component_data.loc[elementName]
            timeStepDuration_ele = timeStepDuration.loc[elementName]

            if year is not None:
                # only for the given year
                timeStepsYear = EnergySystem.encode_time_step(elementName+_storageString,EnergySystem.decode_time_step(None, year, "yearly"),yearly=True)
                totalValue = (component_data*timeStepDuration_ele)[timeStepsYear].sum(axis=1)
            else:
                # for all years
                if split_years:
                    totalValueTemp = pd.DataFrame(index=component_data.index, columns=self.years)
                    for yearTemp in self.years:
                        # set a proxy for the element name
                        timeStepsYear = EnergySystem.encode_time_step(elementName+_storageString,EnergySystem.decode_time_step(None, yearTemp, "yearly"),yearly=True)
                        totalValueTemp[yearTemp] = (component_data*timeStepDuration_ele)[timeStepsYear].sum(axis=1)
                    totalValue = totalValueTemp
                else:
                    totalValue = (component_data*timeStepDuration_ele).sum(axis=1)

        # if we do not have an element name
        else:
            totalValue  = component_data.apply(lambda row: row*timeStepDuration.loc[row.name[0]],axis=1)
            if year is not None:
                # set a proxy for the element name
                elementName_proxy = component_data.index.get_level_values(level=0)[0]
                timeStepsYear = EnergySystem.encode_time_step(elementName_proxy+_storageString,
                                                            EnergySystem.decode_time_step(None, year, "yearly"),
                                                            yearly=True)
                totalValue = totalValue[timeStepsYear].sum(axis=1)
            else:
                if split_years:
                    totalValueTemp = pd.DataFrame(index=totalValue.index,columns=self.years)
                    for yearTemp in self.years:
                        # set a proxy for the element name
                        elementName_proxy = component_data.index.get_level_values(level=0)[0]
                        timeStepsYear = EnergySystem.encode_time_step(elementName_proxy + _storageString,EnergySystem.decode_time_step(None, yearTemp, "yearly"),yearly=True)
                        totalValueTemp[yearTemp] = totalValue[timeStepsYear].sum(axis=1)
                    totalValue = totalValueTemp
                else:
                    totalValue = totalValue.sum(axis=1)
        return totalValue

    def _get_ts_duration(self, scenario=None, is_storage = False):
        """ extracts the time steps duration """
        # extract the right timestep duration
        if self.has_scenarios:
            if scenario is None:
                raise ValueError("Please specify a scenario!")
            else:
                if is_storage:
                    timeStepDuration = self.timeStepStorageDuration[scenario].unstack()
                else:
                    timeStepDuration = self.timeStepOperationalDuration[scenario].unstack()
        else:
            if is_storage:
                timeStepDuration = self.timeStepStorageDuration.unstack()
            else:
                timeStepDuration = self.timeStepOperationalDuration.unstack()
        return timeStepDuration

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
            # set the timesteps
            EnergySystem.set_sequence_time_steps_dict(self.results["dictSequenceTimeSteps"])
            component_name = component.name
            component_data = component.unstack()
        else:
            raise TypeError(f"Type {type(component).__name__} of input is not supported.")

        return component_name,component_data

    def _get_ts_type(self, component_data,component_name):
        """ get time step type (operational, storage, yearly) """
        _headerOperational = self.results["analysis"]["header_data_inputs"]["set_time_steps_operation"]
        _headerStorage = self.results["analysis"]["header_data_inputs"]["set_time_steps_storage_level"]
        _headerYearly = self.results["analysis"]["header_data_inputs"]["set_time_steps_yearly"]
        if component_data.columns.name == _headerOperational:
            return "operational"
        elif component_data.columns.name == _headerStorage:
            return "storage"
        elif component_data.columns.name == _headerYearly:
            return "yearly"
        else:
            raise KeyError(f"Column index name of '{component_name}' ({component_data.columns.name}) is unknown. Should be (operational, storage, yearly)")

    def _get_hours_of_year(self,year):
        """ get total hours of year """
        _total_hours_per_year = self.results["system"]["unaggregated_time_steps_per_year"]
        _hours_of_year = list(range(year*_total_hours_per_year,(year+1)*_total_hours_per_year))
        return _hours_of_year

    def __str__(self):
        return f"Results of '{self.path}'"

