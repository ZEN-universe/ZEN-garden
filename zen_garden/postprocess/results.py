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
import shutil
import json
import zlib
import os

from ..model.objects.energy_system import EnergySystem
from ..model.objects.parameter import Parameter
from ..utils import RedirectStdStreams

class Postprocess:

    def __init__(self, model, **kwargs):
        """postprocessing of the results of the optimization
        :param model: optimization model
        :param **kwargs: Additional keyword arguments such as the model name used for the directory to save the
                         results in"""

        # get name or directory
        self.modelName = kwargs.get('modelName', "")
        self.nameDir = kwargs.get('nameDir', os.path.join('./outputs', self.modelName))

        # get the necessary stuff from the model
        self.model = model.model
        self.system = model.system
        self.analysis = model.analysis
        self.solver = model.solver
        self.opt = model.opt
        self.params = Parameter.getParameterObject()

        # get the compression param
        self.compress = self.system["compressOutput"]

        # create the output directory
        os.makedirs(self.nameDir, exist_ok=True)

        # save the pyomo yml
        if self.system["writeResultsYML"]:
            with RedirectStdStreams(open(os.path.join(self.nameDir, "results.yml"), "w+")):
                model.results.write()

        # save everything
        self.saveParam()
        self.saveVar()
        self.saveSystem()
        self.saveAnalysis()
        self.saveSolver()
        self.saveOpt()

        # extract and save sequence time steps, we transform the arrays to lists
        self.dictSequenceTimeSteps = self.flatten_dict(EnergySystem.getSequenceTimeStepsDict())
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
        if not self.compress and sys.getsizeof(serialized_dict)/1024**2 > self.system["maxOutputSizeMB"]:
            print(f"WARNING: The file {name}.json would be larger than the maximum allowed output size of "
                  f"{self.system['maxOutputSizeMB']}MB, compressing...")
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
        for param in self.params.parameterList:
            # get the values
            vals = getattr(self.params, param)
            # create a dictionary if necessary
            if not isinstance(vals, dict):
                indices = [(0, )]
                data = [vals]
            # if the returned dict is emtpy we create a None value
            elif len(vals) == 0:
                indices = [(0,)]
                data = [None]
            else:
                indices = [k if isinstance(k, tuple) else (k, ) for k in vals.keys()]
                data = [v for v in vals.values()]

            # create dataframe
            df = pd.DataFrame(data=data, columns=["value"], index=pd.MultiIndex.from_tuples(indices))

            # update dict
            data_frames[param] = {"dataframe": {f"{k}": v for k, v in df.to_dict(orient="index").items()},
                                  "docstring": self.params.docs[param]}

        # write to json
        self.write_file(os.path.join(self.nameDir, 'paramDict'), data_frames)

    def saveVar(self):
        """ Saves the variable values to a json file which can then be
        post-processed immediately or loaded and postprocessed at some other time"""

        # dataframe serialization
        data_frames = {}
        for var in self.model.component_objects(pe.Var, active=True):
            # get indices and values
            indices = [index if isinstance(index, tuple) else (index, ) for index in var]
            values = [getattr(var[index], "value", None) for index in indices]

            # create dataframe
            df = pd.DataFrame(data=values, columns=["value"], index=pd.MultiIndex.from_tuples(indices))

            # save to dict, we transform the multiindex tuples to strings such that we can use standard json to dump
            data_frames[var.name] = {"dataframe": {f"{k}": v for k, v in df.to_dict(orient="index").items()},
                                     "docstring": var.doc}

        self.write_file(os.path.join(self.nameDir, 'varDict'), data_frames)

    def saveSystem(self):
        """
        Saves the system dict as json
        """
        self.write_file(os.path.join(self.nameDir, 'System'), self.system)

    def saveAnalysis(self):
        """
        Saves the analysis dict as json
        """
        self.write_file(os.path.join(self.nameDir, 'Analysis'), self.analysis)

    def saveSolver(self):
        """
        Saves the solver dict as json
        """
        self.write_file(os.path.join(self.nameDir, 'Solver'), self.solver)

    def saveOpt(self):
        """
        Saves the opt dict as json
        """
        self.write_file(os.path.join(self.nameDir, 'optDict'), self.opt.__dict__)

        # copy the log file
        shutil.copy2(os.path.abspath(self.opt._log_file), self.nameDir)

    def saveSequenceTimeSteps(self):
        """
        Saves the dictAllSequenceTimeSteps dict as json
        """
        self.write_file(os.path.join(self.nameDir, 'dictAllSequenceTimeSteps'), self.dictSequenceTimeSteps)

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


class Results(object):
    """
    This class reads in the results after the pipeline has run
    """

    # TODO: if option to iterate
    def __init__(self, path, expand=True, load_opt=False):
        """
        Initializes the Results class with a given path
        :param path: Path to the output of the optimization problem
        :param expand: Expand the path to all scenarios via glob, i.e. path*
        """

        # TODO: implement expansion
        self.path = os.path.abspath(path)

        # check if the path exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"No such file or directory: {self.path}")

        # load everything
        self.dictParam = self.load_params(self.path)
        self.dictVar = self.load_vars(self.path)
        self.system = self.load_system(self.path)
        self.analysis = self.load_analysis(self.path)
        self.solver = self.load_solver(self.path)
        self.dictSequenceTimeSteps = self.load_sequence_time_steps(self.path)

        # the opt we only load when requested
        if load_opt:
            self.optdict = self.load_opt(self.path)

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
                                                     orient="index")

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
        raw_dict = cls._read_file(os.path.join(path, "dictAllSequenceTimeSteps"))
        dictSequenceTimeSteps = json.loads(raw_dict)

        return dictSequenceTimeSteps

    def __str__(self):
        return f"Results of '{self.path}'"
