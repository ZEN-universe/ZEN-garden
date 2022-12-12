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
from ..model.objects.component import Parameter, Variable, Constraint
from ..utils import RedirectStdStreams


class Postprocess:

    def __init__(self, model, scenarios, modelName, subfolder=None, scenario_name=None):
        """postprocessing of the results of the optimization
        :param model: optimization model
        :param modelName: The name of the model used to name the output folder
        :param subfolder: The subfolder used for the results
        :param scenario_name: The name of the current scenario
        """

        # get the necessary stuff from the model
        self.model = model.model
        self.scenarios = scenarios
        self.system = model.system
        self.analysis = model.analysis
        self.solver = model.solver
        self.opt = model.opt
        self.params         = Parameter.getComponentObject()
        self.vars           = Variable.getComponentObject()
        self.constraints    = Constraint.getComponentObject()

        # get name or directory
        self.modelName = modelName
        self.nameDir = pathlib.Path(self.analysis["folderOutput"]).joinpath(self.modelName)

        # deal with the subfolder
        self.subfolder = subfolder
        # here we make use of the fact that None and "" both evaluate to False but any non-empty string doesn't
        if self.subfolder:
            self.nameDir = self.nameDir.joinpath(self.subfolder)
        # create the output directory
        os.makedirs(self.nameDir, exist_ok=True)

        # check if we should overwrite output
        self.overwirte = self.system["overwriteOutput"]
        # get the compression param
        self.compress = self.analysis["compressOutput"]

        # save the pyomo yml
        if self.analysis["writeResultsYML"]:
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
        self.dictSequenceTimeSteps = self.flatten_dict(EnergySystem.getSequenceTimeStepsDict())
        self.saveSequenceTimeSteps(scenario=scenario_name)

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
        if not self.compress and sys.getsizeof(serialized_dict)/1024**2 > self.analysis["maxOutputSizeMB"]:
            print(f"WARNING: The file {name}.json would be larger than the maximum allowed output size of "
                  f"{self.analysis['maxOutputSizeMB']}MB, compressing...")
            force_compression = True

        # prep output file
        if self.compress or force_compression:
            # compress
            f_name = f"{name}.gzip"
            f_mode = "wb"
            serialized_dict = zlib.compress(serialized_dict.encode())
        else:
            # write normal json
            f_name = f"{name}.json"
            f_mode = "w+"

        # write if necessary
        if self.overwirte or not os.path.exists(f_name):
            with open(f_name, f_mode) as outfile:
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
            indexList = self.getIndexList(doc)
            if len(indexList) == 0:
                indexNames = None
            elif len(indexList) == 1:
                indexNames = indexList[0]
            else:
                indexNames = indexList
            # create a dictionary if necessary
            if not isinstance(vals, dict):
                indices = pd.Index(data=[0],name=indexNames)
                data = [vals]
            # if the returned dict is emtpy we create a nan value
            elif len(vals) == 0:
                if len(indexList)>1:
                    indices = pd.MultiIndex(levels=[[]]*len(indexNames),codes=[[]]*len(indexNames),names=indexNames)
                else:
                    indices = pd.Index(data=[],name=indexNames)
                data = []
            # we read out everything
            else:
                indices = list(vals.keys())
                data = list(vals.values())

                # create a multi index if necessary
                if len(indices)>=1 and isinstance(indices[0],tuple):
                    if len(indexList) == len(indices[0]):
                        indices = pd.MultiIndex.from_tuples(indices,names=indexNames)
                    else:
                        indices = pd.MultiIndex.from_tuples(indices)
                else:
                    if len(indexList) == 1:
                        indices = pd.Index(data=indices,name=indexNames)
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
                indexList = self.getIndexList(doc)
                if len(indexList) == 0:
                    indexNames = None
                elif len(indexList) == 1:
                    indexNames = indexList[0]
                else:
                    indexNames = indexList
            else:
                indexList = []
                doc = None
            # get indices and values
            indices = [index for index in var]
            values = [getattr(var[index], "value", None) for index in indices]

            # create a multi index if necessary
            if len(indices)>=1 and isinstance(indices[0], tuple):
                if len(indexList) == len(indices[0]):
                    indices = pd.MultiIndex.from_tuples(indices, names=indexNames)
                else:
                    indices = pd.MultiIndex.from_tuples(indices)
            else:
                if len(indexList) == 1:
                    indices = pd.Index(data=indices, name=indexNames)
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
        self.write_file(fname, self.solver)

    def saveOpt(self):
        """
        Saves the opt dict as json
        """
        if self.solver["name"] != "gurobi_persistent":
            self.write_file(self.nameDir.joinpath('optDict'), self.opt.__dict__)

        # copy the log file
        shutil.copy2(os.path.abspath(self.opt._log_file), self.nameDir)

    def saveSequenceTimeSteps(self, scenario=None):
        """
        Saves the dictAllSequenceTimeSteps dict as json
        """
        # add the scenario name
        if scenario is not None:
            add_on = f"_{scenario}"
        else:
            add_on = ""

            # This we only need to save once
        if self.subfolder:
            fname = self.nameDir.parent.joinpath(f'dictAllSequenceTimeSteps{add_on}')
        else:
            fname = self.nameDir.joinpath(f'dictAllSequenceTimeSteps{add_on}')

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
        indexList = string.split(",")
        indexListFinal = []
        for index in indexList:
            if index in self.analysis["headerDataInputs"].keys():
                indexListFinal.append(self.analysis["headerDataInputs"][index])
            else:
                pass
                # indexListFinal.append(index)
        return indexListFinal
