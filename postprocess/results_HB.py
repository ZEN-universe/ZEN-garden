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
from datetime import datetime
from model.objects.energy_system import EnergySystem
import matplotlib.pyplot as plt

#from postprocess.functions.create_dashboard_dictionary import DashboardDictionary

class Postprocess:

    system    = dict()
    varDict   = dict()
    varDf     = dict()
    paramDict = dict()
    paramDf   = dict()
    modelName = str()


    def __init__(self, model=None, **kwargs):
        """postprocessing of the results of the optimization
        :param model:     optimization model
        :param pyoDict:   input data dictionary
        :param modelName: model name used for the directory to save the results in"""

        # self.model     = model.model
        # self.system    = model.system
        # self.analysis  = model.analysis
        self.modelName = kwargs.get('modelName', self.modelName)
        self.nameDir   = f'./outputs/results{self.modelName}/'

        # self.makeDirs()
        # self.getVarValues()
        # self.saveResults()
        # self.plotResults()

        # case where the class is called from the command line
        if model==None:
            self.loadParam()
            self.loadVar()
            self.loadSystem()
            self.loadAnalysis()
            self.process()

        # case where we are called from compile but should not perform the post-processing
        elif model.analysis['postprocess']==False:

            self.model     = model.model
            self.system    = model.system
            self.analysis  = model.analysis

            self.makeDirs()
            self.saveParam()
            self.saveVar()
            self.saveSystem()
            self.saveAnalysis()

        # case where we should run the post-process as normal
        else:

            self.model     = model.model
            self.system    = model.system
            self.analysis  = model.analysis

            self.makeDirs()
            self.saveParam()
            self.saveVar()
            self.saveSystem()
            self.saveAnalysis()
            self.process()

    def makeDirs(self):
        """create results directory"""
        try:
            os.makedirs(self.nameDir)
        except OSError:
            pass

        try:
            os.makedirs(f'{self.nameDir}/plots/')
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

    def saveParam(self):
        """ Saves the Param values to pickle files which can then be
        post-processed immediately or loaded and postprocessed at some other time"""

        # get all the param values from the model and store in a dict
        for param in self.model.component_objects(pe.Param, active=True):
            # sava params in a dict
            self.paramDict[param.name] = dict()
            for index in param:
                self.paramDict[param.name][index] = pe.value(param[index])

        # save the param dict to a pickle
        with open(f'{self.nameDir}params/paramDict.pickle', 'wb') as file:
            pickle.dump(self.paramDict, file, protocol=pickle.HIGHEST_PROTOCOL)

        # save sequence time steps
        dictSequenceTimeSteps = EnergySystem.getSequenceTimeStepsDict()
        # save the param dict to a pickle
        with open(f'{self.nameDir}dictAllSequenceTimeSteps.pickle', 'wb') as file:
            pickle.dump(dictSequenceTimeSteps, file, protocol=pickle.HIGHEST_PROTOCOL)

    def loadParam(self):
        """ Loads the Param values from previously saved pickle files which can then be
        post-processed """
        with open(f'{self.nameDir}params/paramDict.pickle', 'rb') as file:
            self.paramDict = pickle.load(file)

    def getVarValues(self):
        """get the values assigned to each variable"""

        for var in self.model.component_objects(pe.Var, active=True):
            if 'constraint' not in var.name and 'gdp' not in var.name:
                # save vars in a dict
                self.varDict[var.name] = dict()
                for index in var:
                    if var[index].value:
                        self.varDict[var.name][index] = pe.value(var[index])
                # save vars in a DataFrame
                self.createDataframe(var, self.varDict, self.varDf)

    def saveVar(self):
        """ Saves the variable values to pickle files which can then be
        post-processed immediately or loaded and postprocessed at some other time"""

        # get all the variable values from the model and store in a dict
        for var in self.model.component_objects(pe.Var, active=True):
            if 'constraint' not in var.name and 'gdp' not in var.name:
                # save vars in a dict
                self.varDict[var.name] = dict()
                for index in var:
                    try:
                        self.varDict[var.name][index] = var[index].value
                    except:
                        pass

        # save the variable dict to a pickle
        with open(f'{self.nameDir}vars/varDict.pickle', 'wb') as file:
            pickle.dump(self.varDict, file, protocol=pickle.HIGHEST_PROTOCOL)

    def saveVar_HB(self):  # My own function to save the variables in better dicts
        """ Saves the variable values to pickle files which can then be
        post-processed immediately or loaded and postprocessed at some other time"""

        # get all the variable values from the model and store in a dict
        for var in self.model.component_objects(pe.Var, active=True):
            if 'constraint' not in var.name and 'gdp' not in var.name:
                # create a sub-dict for each variable
                self.varDict[var.name] = dict()
                for index in var:
                    # if variable is sub-splitted (e.g. for child-classes) create sub-dicts
                    if type(index) is tuple:
                        try:
                            self.varDict[var.name][index[0]][index[1:]] = var[index].value
                        except KeyError:
                            self.varDict[var.name][index[0]] = dict()
                            self.varDict[var.name][index[0]][index[1:]] = var[index].value
                    # for index in var:
                    else:
                        try:
                            self.varDict[var.name][index] = var[index].value
                        except:
                            pass

        # save the variable dict to a pickle
        with open(f'{self.nameDir}vars/varDict.pickle', 'wb') as file:
            pickle.dump(self.varDict, file, protocol=pickle.HIGHEST_PROTOCOL)

    def loadVar(self):
        """ Loads the variable values from previously saved pickle files which can then be
        post-processed """
        with open(f'{self.nameDir}vars/varDict.pickle', 'rb') as file:
            self.varDict = pickle.load(file)

    def saveSystem(self):
        with open(f'{self.nameDir}System.pickle', 'wb') as file:
            pickle.dump(self.system, file, protocol=pickle.HIGHEST_PROTOCOL)

    def saveAnalysis(self):
        with open(f'{self.nameDir}Analysis.pickle', 'wb') as file:
            pickle.dump(self.analysis, file, protocol=pickle.HIGHEST_PROTOCOL)

    def loadSystem(self):
        """ Loads the system object from previously saved pickle files which can then be
        post-processed """
        with open(f'{self.nameDir}System.pickle', 'rb') as file:
            self.system = pickle.load(file)

    def loadAnalysis(self):
        """ Loads the analysis object from previously saved pickle files which can then be
        post-processed """
        with open(f'{self.nameDir}Analysis.pickle', 'rb') as file:
            self.analysis = pickle.load(file)

    # def createDataframe(self, obj, dict, df):
    #     """ save data in dataframe"""
    #     if dict[obj.name]:
    #         if list(dict[obj.name].keys())[0] == None:
    #             # [index, capacity]
    #             df[obj.name] = pd.DataFrame(dict[obj.name].values(), columns=self.analysis['headerDataOutputs'][obj.name])
    #             self.trimZeros(obj, self.varDf, df[obj.name].columns.values)
    #             print(df)
    #         elif type(list(dict[obj.name].keys())[0]) == int:
    #             # seems like we never come in here
    #             print("DID SOMETHING COME IN HERE??")
    #             df[obj.name] = pd.DataFrame(dict[obj.name].values(), index=list(dict[obj.name].keys()), columns=self.analysis['headerDataOutputs'][obj.name])
    #             self.trimZeros(obj, self.varDf, df[obj.name].columns.values)
    #             print(df)
    #         else:
    #             # [tech, node, time, capacity]
    #             df[obj.name] = pd.DataFrame(dict[obj.name].values(),index=pd.MultiIndex.from_tuples(dict[obj.name].keys())).reset_index()
    #             df[obj.name].columns = self.analysis['headerDataOutputs'][obj.name]
    #             self.trimZeros(obj, self.varDf, df[obj.name].columns.values)
    #             print(df)
    #     else:
    #         print(f'{obj.name} not evaluated in results_HB.py')

    def createDataframe(self, varName, dict, df):
        """ save data in dataframe"""
        if dict[varName]:
            if list(dict[varName].keys())[0] == None:
                # [index, capacity]
                df[varName] = pd.DataFrame(dict[varName].values(), columns=self.analysis['headerDataOutputs'][varName])
                self.trimZeros(varName, self.varDf, df[varName].columns.values)
                print(df)
            elif type(list(dict[varName].keys())[0]) == int:
                # seems like we never come in here
                print("DID SOMETHING COME IN HERE??")
                try:
                    df[varName] = pd.DataFrame(dict[varName].values(), index=list(dict[varName].keys()), columns=self.analysis['headerDataOutputs'][varName])
                except KeyError:
                    logging.info(f"create header for variable {varName}")
                    df[varName] = pd.DataFrame(dict[varName].values(), index=list(dict[varName].keys()))
                self.trimZeros(varName, self.varDf, df[varName].columns.values)
                print(df)
            else:
                # [tech, node, time, capacity]
                df[varName] = pd.DataFrame(dict[varName].values(),index=pd.MultiIndex.from_tuples(dict[varName].keys())).reset_index()
                df[varName].columns = self.analysis['headerDataOutputs'][varName]
                self.trimZeros(varName, self.varDf, df[varName].columns.values)
                print(df)
        else:
            print(f'{varName} not evaluated in results.py')

    def trimZeros(self, varName, df, c=[0]):
        """ Trims out the zero rows in the dataframe """
        df[varName] = df[varName].loc[~(df[varName][c[-1]]==0)]

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

    def process(self):
        print(self.varDict.items())
        for var,dic in self.varDict.items():
            self.createDataframe(var, self.varDict, self.varDf)
            self.varDf[var].to_csv(f'{self.nameDir}vars/{var}.csv', index=False)
            self.plotResults()

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
            elif varName=='carrierFlowCharge' or varName=='carrierFlowDischarge' or varName=='levelCharge':
                print('not implemented')
            else: # --> 3)
                c = df.columns
                t = 0
                labels = df[c[0]].unique()

                df = df.sort_values(by=['node',c[0]])
                df = df.set_index(['node',c[0],'time'])
                df.loc[(slice(None),slice(None),t), :].reset_index(level=['time'],drop=['True']).unstack().plot(kind='bar', stacked=True, title=varName+'\ntimeStep='+str(t))

                hand, lab = plt.gca().get_legend_handles_labels()
                leg = []
                for l in lab:
                    for la in labels:
                        if la in l:
                            leg.append(la)
                plt.legend(leg)

                path = f'{self.nameDir}plots/'+varName+'.png'
                plt.savefig(path)

        # safe dictSequenceTimeSteps
        dictAllSequenceTimeSteps = EnergySystem.getSequenceTimeStepsDict()
        with open(f'{self.nameDir}dictAllSequenceTimeSteps.pickle', 'wb') as file:
            pickle.dump(dictAllSequenceTimeSteps, file, protocol=pickle.HIGHEST_PROTOCOL)
    # indexNames  = self.getProperties(getattr(self.model, varName).doc)
    # self.varDf[varName] = pd.DataFrame(varResults, index=pd.MultiIndex.from_tuples(indexValues, names=indexNames))

if __name__ == "__main__":
    today = datetime.date()
    modelName = "model_" + today.strftime("%Y-%m-%d")
    evaluation = Postprocess(modelName=modelName)