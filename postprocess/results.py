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
        self.analysis  = model.analysis
        self.modelName = kwargs.get('modelName', self.modelName)
        self.nameDir   = f'./outputs/results{self.modelName}/'

        self.makeDirs()
        self.getVarValues()
        #self.getParamValues()
        self.saveResults()
        self.plotResults()

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
            if list(dict[obj.name].keys())[0] == None:
                # [index, capacity]
                df[obj.name] = pd.DataFrame(dict[obj.name].values(), columns=self.analysis['headerDataOutputs'][obj.name])
                self.trimZeros(obj, self.varDf, df[obj.name].columns.values)
                print(df)
            elif type(list(dict[obj.name].keys())[0]) == int:
                # seems like we never come in here
                print("DID SOMETHING COME IN HERE??")
                df[obj.name] = pd.DataFrame(dict[obj.name].values(), index=list(dict[obj.name].keys()), columns=self.analysis['headerDataOutputs'][obj.name])
                self.trimZeros(obj, self.varDf, df[obj.name].columns.values)
                print(df)
            else:
                # [tech, node, time, capacity]
                df[obj.name] = pd.DataFrame(dict[obj.name].values(),index=pd.MultiIndex.from_tuples(dict[obj.name].keys())).reset_index()
                df[obj.name].columns = self.analysis['headerDataOutputs'][obj.name]
                self.trimZeros(obj, self.varDf, df[obj.name].columns.values)
                print(df)
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

    def plotResults(self):
        for varName, df in self.varDf.items():
            # Need to catch here the empty dataframes because we cant plot something that isnt there
            if df.empty:
                continue
            elif varName=='installTechnology':    # --> 1)
                print('not implemented')
            elif varName=='carrierFlow' or varName=='carrierLoss': # --> 2)
                print('not implemented')
            elif varName=='dependentFlowApproximation' or varName=='inputFlow' or varName=='outputFlow' or varName=='referenceFlowApproximation': # --> 4)
                print('not implemented')
            elif varName=='carbonEmissionsCarrierTotal' or varName=='capexTotal' or varName=='carbonEmissionsTechnologyTotal' or varName=='carbonEmissionsTotal' or varName=='costCarrierTotal' or varName=='opexTotal':
                print('not implemented')
            else: # --> 3)
                # print(varName)
                c = df.columns
                df2 = pd.DataFrame({c[0]:['biomass','biomass','biomass'],"node":['IT','IT','IT'],"time":[0,1,2],"capacity[GWh]":[5000000,5000000,5000000]})
                df = df.append(df2).reset_index(drop=True)
                print(df)
                df = df.sort_values(by=['node',c[0]])
                print(df)
                # print(c)
                # df=df.set_index(['node', c[0],'time'])
                # df = df.set_index(['node',c[0],'time'])
                df = df.set_index(['node',c[0]])
                df = df.loc[df['time']==0,"capacity[GWh]"]
                print(df)
                data = df.value
                # df = df.loc[(slice(None),slice(None),0), :].reset_index(level=['time'],drop=['True'])
                # print(df)
                # print(df.loc[(slice(None),slice(None),0), :])
                # print(df.loc[df['time']==0,[c[0],'capacity[GWh]']])
                # ax = df.loc[(slice(None),slice(None),0), :].reset_index(level=[c[0],'time'],drop=['False', 'True']).plot.bar(rot=0)
                # df.loc[(slice(None),slice(None),0), :].reset_index(level=['time'],drop=['True']).unstack().plot(kind='bar', stacked=True)
                # data = df.loc[(slice(None),slice(None),0), :].reset_index(level=['time'],drop=['True']).value
                data.unstack().plot(kind='bar', stacked=True)
                # ax = df.loc[df['time']==0,[c[0],'capacity[GWh]']].plot.bar(rot=0)
                plt.show()

            # print(varName)
            # print(df)

    # indexNames  = self.getProperties(getattr(self.model, varName).doc)
    # self.varDf[varName] = pd.DataFrame(varResults, index=pd.MultiIndex.from_tuples(indexValues, names=indexNames))

    if __name__ == "__main__":
        today = datetime.date()
        filename = "model_" + today.strftime("%Y-%m-%d")
        with open("./outputs/{filename}.pkl", 'rb') as inp:
            tech_companies = pickle.load(inp)

        evaluation = Postprocess(model, modelName='test')
