"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
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
                    self.varDict[var.name][index] = pe.value(var[index])
                # save vars in a DataFrame
                self.createDataframe(var, self.varDict, self.varDf)

    def createDataframe(self, obj, dict, df):
        """ save data in dataframe"""
        if dict[obj.name]:

            if list(dict[obj.name].keys())[0] == None:
                # [index, capacity]
                # type 2, and 4 comes in here
                if obj.name=='capexTotal' or obj.name=='costCarrierTotal' or obj.name=='opexTotal':
                    df[obj.name] = pd.DataFrame(dict[obj.name].values(), columns=['capacity[k€]'])
                else:   # obj.name=='carbonEmissionsCarrierTotal' or obj.name=='carbonEmissionsTechnologyTotal' or obj.name=='carbonEmissionsgTotal'
                    df[obj.name] = pd.DataFrame(dict[obj.name].values(), columns=['capacity[GWh]'])

                self.trimZeros(obj, self.varDf, df[obj.name].columns.values)

            elif type(list(dict[obj.name].keys())[0]) == int:
                # seems like we never come in here
                print("DID SOMETHING COME IN HERE??")
                df[obj.name] = pd.DataFrame(dict[obj.name].values(),
                                            index=list(dict[obj.name].keys()))
                self.trimZeros(obj, self.varDf)
            else:
                # [tech, node, time, capacity]
                # both type 1, 3 and 5 come in here
                if obj.name=='carbonEmissionsCarrier' or obj.name=='costCarrier' or obj.name=='exportCarrierFlow' or obj.name=='importCarrierFlow':
                    df[obj.name] = pd.DataFrame(dict[obj.name].values(),
                                            index=pd.MultiIndex.from_tuples(dict[obj.name].keys())).reset_index()
                    df[obj.name].columns = ['carrier','node','time','capacity[GWh]']
                elif obj.name=='carrierFlow' or obj.name=='carrierLoss':
                    df[obj.name] = pd.DataFrame(dict[obj.name].values(),
                                            index=pd.MultiIndex.from_tuples(dict[obj.name].keys())).reset_index()
                    df[obj.name].columns=['trans_tech','n1->n2','time','capacity[GWh]']
                elif obj.name=='dependentFlowApproximation' or obj.name=='inputFlow' or obj.name=='outputFlow' or obj.name=='referenceFlowApproximation':
                    df[obj.name] = pd.DataFrame(dict[obj.name].values(),
                                            index=pd.MultiIndex.from_tuples(dict[obj.name].keys())).reset_index()
                    df[obj.name].columns=['conv_tech','carrier','node','time','capacity[GWh]']
                elif obj.name=='installTechnology':
                    df[obj.name] = pd.DataFrame(dict[obj.name].values(),
                                            index=pd.MultiIndex.from_tuples(dict[obj.name].keys())).reset_index()
                    df[obj.name].columns=['conv_tech','node','time','T/F']
                else:
                    df[obj.name] = pd.DataFrame(dict[obj.name].values(),
                                            index=pd.MultiIndex.from_tuples(dict[obj.name].keys())).reset_index()
                    df[obj.name].columns=['conv_tech','node','time','capacity[GWh]']

                self.trimZeros(obj, self.varDf, df[obj.name].columns.values)
        else:
            print(f'{obj.name} not evaluated in results.py')

    def trimZeros(self, obj, df, c=[0]):
        """ Trims out the zero rows in the dataframe """
        df[obj.name] = df[obj.name].loc[~(df[obj.name][c[-1]]==0)]

        # TODO: handle the case where you are left with an empty dataframe
        # --> maybe put the check in saveResults() and either have no csv for
        #   empty dataframe or create a list to keep track of which variables are empty

    def saveResults(self):
        """save the input data (paramDict, paramDf) and the results (varDict, varDf)"""

        # Save parameter data
        with open(f'{self.nameDir}params/paramDict.pickle', 'wb') as file:
            pickle.dump(self.paramDict, file, protocol=pickle.HIGHEST_PROTOCOL)
        for paramName, df in self.paramDf.items():
            df.to_csv(f'{self.nameDir}params/{paramName}.csv')

        # Save variable data
        with open(f'{self.nameDir}vars/varDict.pickle', 'wb') as file:
            pickle.dump(self.varDict, file, protocol=pickle.HIGHEST_PROTOCOL)
        for varName, df in self.varDf.items():
            df.to_csv(f'{self.nameDir}vars/{varName}.csv', index=False)

    # indexNames  = self.getProperties(getattr(self.model, varName).doc)
    # self.varDf[varName] = pd.DataFrame(varResults, index=pd.MultiIndex.from_tuples(indexValues, names=indexNames))

    if __name__ == "__main__":
        today = datetime.date()
        filename = "model_" + today.strftime("%Y-%m-%d")
        with open("./outputs/{filename}.pkl", 'rb') as inp:
            tech_companies = pickle.load(inp)

        evaluation = Postprocess(model, modelName='test')
