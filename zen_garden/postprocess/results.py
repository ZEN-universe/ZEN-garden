"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Class is defining the postprocessing of the results.
              The class takes as inputs the optimization problem (model) and the system configurations (system).
              The class contains methods to read the results and save them in a result dictionary (resultDict).
==========================================================================================================================================================================="""

import pyomo.environ as pe
import os
import pickle
import pandas as pd
import json

from ..model.objects.energy_system import EnergySystem
from ..model.objects.parameter import Parameter
from ..utils import RedirectStdStreams

class Postprocess:

    def __init__(self, model, **kwargs):
        """postprocessing of the results of the optimization
        :param model: optimization model
        :param results: the results instance of the optimization model
        :param **kwargs: Additional keyword arguments such as the model name used for the directory to save the
                         results in"""

        # get name or directory
        self.modelName = kwargs.get('modelName', "")
        self.nameDir = kwargs.get('nameDir', os.path.join('./outputs', self.modelName))

        # get the necessary stuff from the model
        self.model = model.model
        self.system = model.system
        self.analysis = model.analysis
        self.params = Parameter.getParameterObject()

        # create the output directory
        os.makedirs(self.nameDir, exist_ok=True)

        # save the pyomo yml
        with RedirectStdStreams(open(os.path.join(self.nameDir, "results.yml"), "w+")):
            model.results.write()

        # save everything
        self.saveParam()
        self.saveVar()
        self.saveSystem()
        self.saveAnalysis()

        # TODO: Find a better format for this
        # save sequence time steps
        dictSequenceTimeSteps = EnergySystem.getSequenceTimeStepsDict()

        with open(os.path.join(self.nameDir, 'dictAllSequenceTimeSteps.pickle'), 'wb') as file:
            pickle.dump(dictSequenceTimeSteps, file, protocol=pickle.HIGHEST_PROTOCOL)

        # case where we should run the post-process as normal
        if model.analysis['postprocess']:
            pass
            # TODO: implement this...
            #self.process()

    def saveParam(self):
        """ Saves the Param values to pickle files which can then be
        post-processed immediately or loaded and postprocessed at some other time"""

        # get all the param values from the model and store in a dict
        # for param in self.model.component_objects(pe.Param, active=True):
        #     # sava params in a dict
        #     self.paramDict[param.name] = dict()
        #     for index in param:
        #         self.paramDict[param.name][index] = pe.value(param[index])
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
        with open(os.path.join(self.nameDir, 'paramDict.json'), 'w+') as outfile:
            json.dump(data_frames, outfile, indent=2)

    def saveVar(self):
        """ Saves the variable values to pickle files which can then be
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

        with open(os.path.join(self.nameDir, 'varDict.json'), 'w+') as outfile:
            json.dump(data_frames, outfile, indent=2)

    def saveSystem(self):
        """
        Saves the system dict as json
        """
        with open(os.path.join(self.nameDir, 'System.json'), 'w+') as outfile:
            json.dump(self.system, outfile, indent=2)

    def saveAnalysis(self):
        """
        Saves the analysis dict as json
        """
        with open(os.path.join(self.nameDir, 'Analysis.json'), 'w+') as outfile:
            json.dump(self.analysis, outfile, indent=2)


class Results(object):
    """
    This class reads in the results after the pipeline has run
    """

    def __init__(self, path):
        """
        Initializes the Results class with a given path
        :param path: Path to the output of the optimization problem
        """

        self.path = os.path.abspath(path)

        # check if the path exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"No such file or directory: {self.path}")

        # load everything
        self.dictParam = self.load_params(self.path)
        self.dictVar = self.load_vars(self.path)
        self.system = self.load_system(self.path)
        self.analysis = self.load_analysis(self.path)
        self.dictSequenceTimeSteps = self.load_sequence_time_steps(self.path)

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
        with open(os.path.join(path, "paramDict.json"), "r") as f:
            paramDict_raw = json.load(f)

        return cls._dict2df(paramDict_raw)

    @classmethod
    def load_vars(cls, path):
        """
        Loads the var dict from a given path
        :param path: Path to load the var dict from
        :return: The var dict
        """

        # load the raw dict
        with open(os.path.join(path, "varDict.json"), "r") as f:
            varDict_raw = json.load(f)

        return cls._dict2df(varDict_raw)

    @classmethod
    def load_system(cls, path):
        """
        Loads the system dict from a given path
        :param path: Directory to load the dictionary from
        :return: The system dictionary
        """

        with open(os.path.join(path, "System.json"), "r") as f:
            system_dict = json.load(f)

        return system_dict

    @classmethod
    def load_analysis(cls, path):
        """
        Loads the analysis dict from a given path
        :param path: Directory to load the dictionary from
        :return: The analysis dictionary
        """

        with open(os.path.join(path, "Analysis.json"), "r") as f:
            analysis_dict = json.load(f)

        return analysis_dict

    @classmethod
    def load_sequence_time_steps(cls, path):
        """
        Loads the dictSequenceTimeSteps from a given path
        :param path: Path to load the dict from
        :return: dictSequenceTimeSteps
        """

        with open(os.path.join(path, "dictAllSequenceTimeSteps.pickle"), "rb") as f:
            dictSequenceTimeSteps = pickle.load(f)

        return dictSequenceTimeSteps

    def __str__(self):
        return f"Results of '{self.path}'"
    