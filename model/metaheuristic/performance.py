"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      January-2022
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class for the collection of the performance metrics of the metaheuristic.
==========================================================================================================================================================================="""

import numpy as np
import time

class Performance:

    def __init__(self, object):

        self.object = object
        # value used in applying convergence criterion for objective function value
        self.minValue = object.nlpDict['hyperparameters']['minVal']
        self.maxValue = object.nlpDict['hyperparameters']['maxVal']
        # condition for convergence
        self.conditionDelta = self.object.solver['convergenceCriterion']['conditionDelta']

        self.timeRuns = []
        self.optimumRuns = []

        self.VariablesHistoryRuns = {}
        for type in ['R', 'O']:
            self.VariablesHistoryRuns[type] = {}
            for name in self.object.dictVars[type]['names']:
                self.VariablesHistoryRuns[type][name] = []

        self.initializeRun()

    def initializeRun(self):
        """ method to initialize lists collecting metrics at each run
        """

        self.startTime = time.time()
        self.converged = False
        self.iteration0 = 0
        self.optimum = []
        self.VariablesHistory = {}
        for type in ['R', 'O']:
            self.VariablesHistory[type] = {}
            for name in self.object.dictVars[type]['names']:
                self.VariablesHistory[type][name] = []

    def record(self, solutionInstance):

        # record the optimum for the single iteration
        self.optimum.append(solutionInstance.f[0])
        # record the solution space at the single iteration
        for type in ['R', 'O']:
            for name in self.object.dictVars[type]['names']:
                idx = self.object.dictVars[type]['name_to_idx'][name]
                self.VariablesHistory[type][name].append(solutionInstance.SA[type][0, idx])

    def checkConvergence(self, iteration):

        optimum = np.array(self.optimum[self.iteration0:])
        optimum[(optimum >= 0) & (optimum <= self.minValue)] = self.minValue
        optimum[(optimum <= 0) & (optimum >= -self.minValue)] = -self.minValue
        optimum[optimum >= self.maxValue] = self.maxValue
        optimum[optimum <= -self.maxValue] = -self.maxValue

        # historical absolute improvement: f(t) - f(t-1)
        self.deltaAbsolute = np.abs(optimum[1:] - optimum[:-1])
        # historical relative improvement: (f(t) - f(t-1))/f(t-1)
        self.deltaRelative = np.abs(self.deltaAbsolute / optimum[:-1])

        if self.conditionDelta == 'relative':
            self.delta = self.deltaRelative
        elif self.conditionDelta == 'absolute':
            self.delta = self.deltaAbsolute

        # index of convergence evaluation based on delta optimum
        iterationArray = np.arange(self.delta.size)

        # iterations when stagnation occurs
        iterationStagnationArray = iterationArray[self.delta <= self.object.nlpDict['hyperparameters']['epsilon']]

        if iterationStagnationArray.size > 1:
            # delta iterations between consecutive recordings
            deltaIterations = (iterationStagnationArray[1:] - iterationStagnationArray[:-1])
            indexes = np.arange(deltaIterations.size)
            iterationStartStagnation = 0
            # last index with non-consecutive iterations satisfying stagnation
            if True in (deltaIterations != 1):
                iterationStartStagnation = indexes[deltaIterations != 1][-1] + 1

            # number of iterations satisfying stagnation
            self.stagnationIteration = deltaIterations[iterationStartStagnation:].size

            if self.stagnationIteration >= self.object.nlpDict['hyperparameters']['MaxStagIter']:
                self.converged = True
                self.iteration0 = iteration + 1

    def restart(self, iteration, solutionInstance):
        """method to initialize the solution archive with random values while keeping memory of the
        """
        print(f"\n -- restart at {iteration} --\n")
        self.converged = False

        # store temporarily the optimum found before initialising the solution archive with random values
        SAOptimum = {}
        for type in ['R', 'O']:
            SAOptimum[type] = solutionInstance.SA[type][0, :].copy()

        solutionInstance.solutionSets()

        for type in ['R', 'O']:
            solutionInstance.SA[type][0, :] = SAOptimum[type]

    def newRun(self):

        self.optimumRuns.append(self.optimum[-1])

        # store each variable's last iteration value
        for type in ['R', 'O']:
            for name in self.VariablesHistoryRuns[type].keys():
                self.VariablesHistoryRuns[type][name].append(self.VariablesHistory[type][name][-1])

        self.timeRuns.append(time.time() - self.startTime)

        self.initializeRun()