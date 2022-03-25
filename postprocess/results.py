"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class is defining the postprocessing of the results.
              The class takes as inputs the optimization problem (model) and the system configurations (system).
              The class contains methods to read the results and save them in a result dictionary (resultDict).
==========================================================================================================================================================================="""
import logging
import pyomo.environ as pe
import csv
import os
import pickle
import pandas as pd

#from postprocess.functions.create_dashboard_dictionary import DashboardDictionary

class Postprocess:

    system    = dict()
    varDict   = dict()
    varDf     = dict()
    paramDict = dict()
    paramDf   = dict()
    modelName = str()


    def __init__(self, model, **kwargs):
        """postprocessing of the results of the optimization
        :param model:     optimization model
        :param pyoDict:   input data dictionary
        :param modelName: model name used for the directory to save the results in"""

        self.model     = model.model
        self.system    = model.system
        self.modelName = kwargs.get('modelName', self.modelName)
        self.nameDir   = f'./outputs/results{self.modelName}/'

        self.makeDirs()
        self.getVarValues()
        #self.getParamValues()
        self.saveResults()

    def makeDirs(self):
        """create results directory"""
        try:
            os.makedirs(self.nameDir)
        except OSError:
            pass

        try:
            os.makedirs(f'{self.nameDir}/params/')
            os.makedirs(f'{self.nameDir}/vars/')
        except OSError:
            pass

    def getParamValues(self):
        """get the values assigned to each variable"""

        for param in self.model.component_objects(pe.Param, active=True):
            # sava params in a dict
            self.paramDict[param.name] = dict()
            for index in param:
                self.paramDict[param.name][index] = pe.value(param[index])
            # save params in a dataframe
            self.createDataframe(param, self.paramDict, self.paramDf)

    def getVarValues(self):
        """get the values assigned to each variable"""

        for var in self.model.component_objects(pe.Var, active=True):
            if 'constraint' not in var.name and 'gdp' not in var.name:
                # save vars in a dict
                self.varDict[var.name] = dict()
                for index in var:
                    try:
                        self.varDict[var.name][index] = var[index].value
                    except:
                        pass
                # save vars in a DataFrame
                self.createDataframe(var, self.varDict, self.varDf)

    def createDataframe(self, obj, dict, df):
        """ save data in dataframe"""

        if dict[obj.name]:
            #TODO add names to columns in DF
            if list(dict[obj.name].keys())[0] == None:
                df[obj.name] = pd.DataFrame(dict[obj.name].values())

            elif type(list(dict[obj.name].keys())[0]) == int:
                df[obj.name] = pd.DataFrame(dict[obj.name].values(),
                                            index=list(dict[obj.name].keys()))
            else:
                df[obj.name] = pd.DataFrame(dict[obj.name].values(),
                                            index=pd.MultiIndex.from_tuples(dict[obj.name].keys()))
        else:
            print(f'{obj.name} not evaluated in results.py')


    def saveResults(self):
        """save the input data (paramDict, paramDf) and the results (varDict, varDf)"""

        with open(f'{self.nameDir}params/paramDict.pickle', 'wb') as file:
            pickle.dump(self.paramDict, file, protocol=pickle.HIGHEST_PROTOCOL)
        for paramName, df in self.paramDf.items():
            df.to_csv(f'{self.nameDir}params/{paramName}.csv')

        with open(f'{self.nameDir}vars/varDict.pickle', 'wb') as file:
            pickle.dump(self.varDict, file, protocol=pickle.HIGHEST_PROTOCOL)
        for varName, df in self.varDf.items():
            df.to_csv(f'{self.nameDir}vars/{varName}.csv')

    # indexNames  = self.getProperties(getattr(self.model, varName).doc)
    # self.varDf[varName] = pd.DataFrame(varResults, index=pd.MultiIndex.from_tuples(indexValues, names=indexNames))

    if __name__ == "__main__":
        today = datetime.date()
        filename = "model_" + today.strftime("%Y-%m-%d")
        with open("./outputs/{filename}.pkl", 'rb') as inp:
            tech_companies = pickle.load(inp)

        evaluation = Postprocess(model, modelName='test')

