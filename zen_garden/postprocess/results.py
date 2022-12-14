"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Class is defining to read in the results of an Optimization problem.
==========================================================================================================================================================================="""

import numpy as np
import pandas as pd
import json
import zlib
import os

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
        self.years = list(range(0, self.results["system"]["optimizedYears"]))

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
        elif self.results["system"]["conductScenarioAnalysis"]:
            self.has_scenarios = True
            self.scenarios = [f"scenario_{scenario}" for scenario in self.results["scenarios"].keys()]
        # there are no scenarios
        else:
            self.has_scenarios = False
            self.scenarios = [None]
        # myopic foresight
        if self.results["system"]["useRollingHorizon"]:
            self.has_MF = True
            self.mf = [f"MF_{stepHorizon}" for stepHorizon in self.years]
        else:
            self.has_MF = False
            self.mf = [None]

        # cycle through the dirs
        for scenario in self.scenarios:
            # init dict
            self.results[scenario] = {}

            # load the corresponding timestep dict
            time_dict = self.load_sequence_time_steps(self.path, scenario)
            self.results[scenario]["dictSequenceTimeSteps"] = time_dict
            self.results[scenario]["SequenceTimeStepsDicts"] = SequenceTimeStepsDicts(time_dict)

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

        # load the time step duration, these are normal dataframe calls (dicts in case of scenarios)
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
    def load_sequence_time_steps(cls, path, scenario=None):
        """
        Loads the dictSequenceTimeSteps from a given path
        :param path: Path to load the dict from
        :param scenario: Name of the scenario to load
        :return: dictSequenceTimeSteps
        """
        # get the file name
        fname = os.path.join(path, "dictAllSequenceTimeSteps")
        if scenario is not None:
            fname += f"_{scenario}"

        # get the dict
        raw_dict = cls._read_file(fname)
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

        # select the scenarios
        if scenario is not None:
            scenarios = [scenario]
        else:
            scenarios = self.scenarios

        # loop
        _data = {}
        for scenario in scenarios:
            # we get the timestep dict
            SequenceTimeStepsDicts = self.results[scenario]["SequenceTimeStepsDicts"]

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
                                techProxy = [k for k in self.results[scenario]["dictSequenceTimeSteps"]["operation"].keys()
                                             if "storagelevel" in k.lower()][0]
                            else:
                                techProxy = [k for k in self.results[scenario]["dictSequenceTimeSteps"]["operation"].keys()
                                             if "storagelevel" not in k.lower()][0]
                            # get the timesteps
                            timeStepsYear = SequenceTimeStepsDicts.encodeTimeStep(techProxy,
                                                                                  SequenceTimeStepsDicts.decodeTimeStep(None, year, "yearly"),
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

        # transform all dataframes to pd.Series with the elementName as name
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
        return self.get_df("timeStepsOperationDuration")

    def loadTimeStepStorageDuration(self):
        """
        Loads duration of operational time steps
        """
        return self.get_df("timeStepsStorageLevelDuration", isStorage=True)

    def getFullTS(self, component, elementName=None, year=None, scenario=None):
        """
        Calculates the full timeseries for a given element
        :param component: Either the dataframe of a component as pandas.Series or the name of the component
        :param elementName: The name of the element
        :param scenario: The scenario for with the component should be extracted (only if needed)
        :return: A dataframe containing the full timeseries of the element
        """
        # extract the data
        component_name, component_data = self._get_component_data(component, scenario)
        # timestep dict
        SequenceTimeStepsDicts = self.results[scenario]["SequenceTimeStepsDicts"]

        ts_type = self._get_ts_type(component_data, component_name)

        if ts_type == "yearly":
            if elementName is not None:
                component_data = component_data.loc[elementName]
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
            _storageString = "StorageLevel"

        # calculate the full time series
        _outputTemp = {}
        for row in component_data.index:
            # we know the name
            if elementName:
                _sequenceTimeSteps = SequenceTimeStepsDicts.getSequenceTimeSteps(elementName+_storageString)
            # we extract the name
            else:
                _sequenceTimeSteps = SequenceTimeStepsDicts.getSequenceTimeSteps(row[0]+_storageString)

            # throw together
            _sequenceTimeSteps = _sequenceTimeSteps[np.in1d(_sequenceTimeSteps,list(component_data.columns))]
            _outputTemp[row] = component_data.loc[row,_sequenceTimeSteps].reset_index(drop=True)
            if year is not None:
                if year in self.years:
                    hours_of_year = self._get_hours_of_year(year)
                    _outputTemp[row] = (_outputTemp[row][hours_of_year]).reset_index(drop=True)
                else:
                    print(f"WARNING: year {year} not in years {self.years}. Return component values for all years")

        # concat and return
        outputDf = pd.concat(_outputTemp,axis=0,keys = _outputTemp.keys()).unstack()
        return outputDf

    def getTotal(self, component, elementName=None, year=None, scenario=None, split_years=True):
        """
        Calculates the total Value of a component
        :param component: Either a dataframe as returned from <get_df> or the name of the component
        :param elementName: The element name to calculate the value for, defaults to all elements
        :param year: The year to calculate the value for, defaults to all years
        :param scenario: The scenario to calculate the total value for
        :param split_years: Calculate the value for each year individually
        :return: A dataframe containing the total value with the specified paramters
        """
        # extract the data
        component_name,component_data = self._get_component_data(component, scenario)
        # timestep dict
        SequenceTimeStepsDicts = self.results[scenario]["SequenceTimeStepsDicts"]

        ts_type = self._get_ts_type(component_data,component_name)

        if ts_type == "yearly":
            if elementName is not None:
                component_data = component_data.loc[elementName]
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
            _storageString = "StorageLevel"

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
                timeStepsYear = SequenceTimeStepsDicts.encodeTimeStep(elementName+_storageString,
                                                                      SequenceTimeStepsDicts.decodeTimeStep(None, year, "yearly"),
                                                                      yearly=True)
                totalValue = (component_data*timeStepDuration_ele)[timeStepsYear].sum(axis=1)
            else:
                # for all years
                if split_years:
                    totalValueTemp = pd.DataFrame(index=component_data.index, columns=self.years)
                    for yearTemp in self.years:
                        # set a proxy for the element name
                        timeStepsYear = SequenceTimeStepsDicts.encodeTimeStep(elementName+_storageString,
                                                                              SequenceTimeStepsDicts.decodeTimeStep(None, yearTemp, "yearly"),
                                                                              yearly=True)
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
                timeStepsYear = SequenceTimeStepsDicts.encodeTimeStep(elementName_proxy+_storageString,
                                                                      SequenceTimeStepsDicts.decodeTimeStep(None, year, "yearly"),
                                                                      yearly=True)
                totalValue = totalValue[timeStepsYear].sum(axis=1)
            else:
                if split_years:
                    totalValueTemp = pd.DataFrame(index=totalValue.index,columns=self.years)
                    for yearTemp in self.years:
                        # set a proxy for the element name
                        elementName_proxy = component_data.index.get_level_values(level=0)[0]
                        timeStepsYear = SequenceTimeStepsDicts.encodeTimeStep(elementName_proxy + _storageString,
                                                                              SequenceTimeStepsDicts.decodeTimeStep(None, yearTemp, "yearly"),
                                                                              yearly=True)
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
            component_name = component.name
            component_data = component.unstack()
        else:
            raise TypeError(f"Type {type(component).__name__} of input is not supported.")

        return component_name, component_data

    def _get_ts_type(self, component_data,component_name):
        """ get time step type (operational, storage, yearly) """
        _headerOperational = self.results["analysis"]["headerDataInputs"]["setTimeStepsOperation"]
        _headerStorage = self.results["analysis"]["headerDataInputs"]["setTimeStepsStorageLevel"]
        _headerYearly = self.results["analysis"]["headerDataInputs"]["setTimeStepsYearly"]
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
        _total_hours_per_year = self.results["system"]["unaggregatedTimeStepsPerYear"]
        _hours_of_year = list(range(year*_total_hours_per_year,(year+1)*_total_hours_per_year))
        return _hours_of_year

    def __str__(self):
        return f"Results of '{self.path}'"

