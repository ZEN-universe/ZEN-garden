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
        self.conditionDelta = self.object.nlpDict['hyperparameters']['convergence']['conditionDelta']

        self.timeRuns = []
        self.optimumRuns = []

        self.VariablesHistoryRuns = {}
        for type in ['R', 'O']:
            self.VariablesHistoryRuns[type] = {}
            for name in self.dictVars[type]['names']:
                self.VariablesHistoryRuns[type][name] = []

        self.initialize_run()

    def initializeRun(self):
        """ method ot initialize lists collecting metrics at each run
        """

        self.startTime = time.time()
        self.converged = False
        self.t0 = 0

        self.VariablesHistory = {}
        for type in ['R', 'O']:
            self.VariablesHistory[type] = {}
            for name in self.dictVars[type]['names']:
                self.VariablesHistory[type][name] = []

    def record(self, solutionInstance):

        # record the optimum for the single iteration
        self.optimum.append(solutionInstance.f[0])
        # record the solution space at the single iteration
        for type in ['R', 'O']:
            for name in self.dictVars[type]['names']:
                idx = self.dictVars[type]['names_to_idx'][name]
                self.VariablesHistory[type][name].append(self.solutionInstance.SA[type][0, idx])

    def checkConvergence(self, t):

        optimum = np.array(self.optimum[self.t0:])
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
        t_iter = np.arange(self.delta.size)

        # iterations when stagnation occurs
        t_stag_iter = t_iter[self.delta <= self.object.nlpDict['hyperparameters']['epsilon']]

        if t_stag_iter.size > 1:
            # delta iterations between consecutive recordings
            delta_t = (t_stag_iter[1:] - t_stag_iter[:-1])
            indexes = np.arange(delta_t.size)
            iter_start_stag = 0
            # last index with non-consecutive iterations satisfying stagnation
            if True in (delta_t != 1):
                iter_start_stag = indexes[delta_t != 1][-1] + 1

            # number of iterations satisfying stagnation
            self.stagnationIteration = delta_t[iter_start_stag:].size

            if self.stagnationIteration >= self.object.nlpDict['hyperparameters']['MaxStagIter']:
                self.converged = True
                self.t0 = t + 1