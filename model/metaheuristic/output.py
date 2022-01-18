"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      January-2022
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining the methods to print the data stored in performance.
==========================================================================================================================================================================="""

import os
import shutil

class Output:

    def __init__(self, object, performanceInstance):

        self.object = object
        self.performanceInstance = performanceInstance

        self.folderOut = './/outputs//'
        self.createFolder(self.folderOut)

        self.folder = self.folderOut + '//Master//'
        self.createFolder(self.folder)

        self.folderLog = self.folder + '//Log//'
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
            shutil.rmtree(folderName)
            os.makedirs(folderName)

    def reportConvergence(self, run, iteration, solutionInstance):

        text = "\n"
        text += f" -- converged at iter {iteration} --" + "\n"
        text += "    delta " + str(self.conditionDelta) + ": " + str(self.performanceInstance.delta[-1]) + "\n"
        text += "    stagnation iterations: " + str(self.performanceInstance.stagnationIteration) + "\n"
        text += "    optimum: " + str(self.performanceInstance.optimum[-1]) + "\n"
        text += "    R - Continuous variables " + "\n"
        for name in self.dictVars['R']['names']:
            idx = self.dictVars['R']['name_to_idx'][name]
            text += "      -> " + name + ": " + str(solutionInstance.SA['R'][0, idx]) + "\n"
        text += "    O - Ordinal variables " + "\n"
        for name in self.dictVars['O']['names']:
            idx = self.dictVars['O']['name_to_idx'][name]
            text += "      -> " + name + ": " + str(solutionInstance.SA['O'][0, idx]) + " " \
                    + str(self.dictVars['O']['values'][0, idx]) + "\n"
        # print to console
        print(text)

        # print to external file
        filename = self.folder_log + "convergence_run{}_iter{}.txt".format(run, iteration)
        f = open(filename, "w+")
        f.write(text)
        f.close()