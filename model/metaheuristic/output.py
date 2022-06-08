"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      January-2022
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Class defining the methods to print the data stored in performance.
==========================================================================================================================================================================="""

import os
import shutil
import pandas as pd
import numpy as np

class Output:

    def __init__(self, object, performanceInstance):

        self.object = object
        self.performanceInstance = performanceInstance

        self.folderOut = './/outputs//'
        self.createFolder(self.folderOut)

        self.folder = self.folderOut + '//master//'
        self.createFolderDeleting(self.folder)

        self.folderLog = self.folder + '//log//'
        self.createFolderDeleting(self.folderLog)

    def createFolder(self, folderName):

        try:
            os.makedirs(folderName)
        except OSError:
            pass

    def createFolderDeleting(self, folderName):

        try:
            os.makedirs(folderName)
        except OSError:
            print(folderName)
            shutil.rmtree(folderName)
            os.makedirs(folderName)

    def reportConvergence(self, run, iteration, solutionInstance):

        text = "\n"
        text += f" -- converged at iter {iteration} --" + "\n"
        text += "    delta " + str(self.performanceInstance.conditionDelta) + ": " + str(self.performanceInstance.delta[-1]) + "\n"
        text += "    stagnation iterations: " + str(self.performanceInstance.stagnationIteration) + "\n"
        text += "    optimum: " + str(self.performanceInstance.optimum[-1]) + "\n"
        text += "    R - Continuous variables " + "\n"
        for name in self.object.dictVars['R']['names']:
            idx = self.object.dictVars['R']['name_to_idx'][name]
            text += "      -> " + name + ": " + str(solutionInstance.SA['R'][0, idx]) + "\n"
        text += "    O - Ordinal variables " + "\n"
        for name in self.object.dictVars['O']['names']:
            idx = self.object.dictVars['O']['name_to_idx'][name]
            text += "      -> " + name + ": " + str(solutionInstance.SA['O'][0, idx]) + " " \
                    + str(self.object.dictVars['O']['values'][0, idx]) + "\n"
        # print to console
        print(text)

        # print to external file
        filename = self.folderLog + "convergence_run{}_iter{}.txt".format(run, iteration)
        f = open(filename, "w+")
        f.write(text)
        f.close()

    def maxFunctionEvaluationsAchieved(self):

        FEMax = self.object.nlpDict['hyperparameters']['FEsMax']
        print(f"\n -- max function evaluations: FEs = {FEMax} --")

    def roundToMinValue(self, values):
        # compute order of magnitude of minimum value accepted
        ndec = np.abs(np.int(np.log10(self.performanceInstance.minValue)))

        return np.round(np.array(values), ndec)

    def fileRun(self, run):
        values = self.roundToMinValue(self.performanceInstance.optimum)
        data = {
            'folder': self.folder + 'runs//',
            'file_name': '{}-optimum'.format(run),
            'file_format': '.csv',
            'values': [values],
            'columns': ['optimum'],
            'name': 'iteration'
        }

        self.createFile(data)

        for type in ['R', 'O']:
            values = []
            keys = []
            for key in self.performanceInstance.VariablesHistory[type].keys():
                values.append(self.performanceInstance.VariablesHistory[type][key])
                keys.append(key)
            values = self.roundToMinValue(values)

            data = {
                'folder': self.folder + 'runs//variables//',
                'file_name': f'{run}-{type}',
                'file_format': '.csv',
                'values': values,
                'columns': keys,
                'name': 'iteration'
            }

            self.createFile(data)

    def createFile(self, data):

        self.createFolder(data['folder'])

        df = pd.DataFrame(data=data['values']).T
        df.columns = data['columns']
        df.index.name = data['name']

        df.to_csv(data['folder'] + data['file_name'] + data['file_format'], header=True, index=True)

    def fileRuns(self):

        values = self.roundToMinValue(self.performanceInstance.optimumRuns)
        data = {
        'folder': self.folder,
        'file_name': 'optimum',
        'file_format': '.csv',
        'values': [values],
        'columns': ['optimum'],
        'name': 'run'
        }

        self.createFile(data)

        for type in ['R', 'O']:
            values = []
            keys = []
            for key in self.performanceInstance.VariablesHistoryRuns[type].keys():
                values.append(self.performanceInstance.VariablesHistoryRuns[type][key])
                keys.append(key)
            values = self.roundToMinValue(values)

            data = {
            'folder': self.folder + 'Variables//',
            'file_name': type,
            'file_format': '.csv',
            'values': values,
            'columns': keys,
            'name': 'run'
            }

            self.createFile(data)

    def reportRuns(self):

        print("-- runs completed --" + "\n")
        text = ""
        text += "     -> optimum" + "\n"
        text += "        mean: " + str(np.mean(self.performanceInstance.optimumRuns).round(2)) + "\n"
        text += "        std: " + str(np.std(self.performanceInstance.optimumRuns).round(2)) + "\n"
        text += "        max: " + str(np.max(self.performanceInstance.optimumRuns).round(2)) + "\n"
        text += "        min: " + str(np.min(self.performanceInstance.optimumRuns).round(2)) + "\n"
        text += "    -> execution time [s]" + "\n"
        text += "        mean: " + str(np.mean(self.performanceInstance.timeRuns).round(2)) + "\n"
        text += "        std: " + str(np.std(self.performanceInstance.timeRuns).round(2)) + "\n"
        text += "        max: " + str(np.max(self.performanceInstance.timeRuns).round(2)) + "\n"
        text += "        min: " + str(np.min(self.performanceInstance.timeRuns).round(2)) + "\n"

        # print to console
        print(text)

        # print to external file
        filename = self.folderLog + "runs.txt"
        f = open(filename, "w+")
        f.write(text)
        f.close()