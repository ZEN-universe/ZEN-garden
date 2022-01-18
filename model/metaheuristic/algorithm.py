"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      January-2022
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class defining the metaheuristic algorithm.
              The algorithm takes as input the set of decision variables subject to nonlinearities and the respective domains.
              Iteratively the predicted values of the decision variables are modified and only those associated to the highest quality solutions are selected.
              The values of the decision variables are then passed to the dictionary of the MILP solver as constant parameters in the model.
==========================================================================================================================================================================="""
import logging
from model.metaheuristic.variables import Variables
from model.metaheuristic.solutions import Solutions
from model.metaheuristic.performance import Performance
from model.metaheuristic.output import Output

class Metaheuristic:

    def __init__(self, model, nlpDict):

        logging.info('initialize metaheuristic algorithm class')

        # instantiate analysis
        self.analysis = model.analysis
        # instantiate system
        self.system = model.system
        # instantiate dictionary containing all the required hyperparameters and input data
        self.nlpDict = nlpDict
        # instantiate the model class
        self.model = model

        # collect the properties of the decision variables handled by the metaheuristic and create a new attribute in
        # the Pyomo dictionary
        self.dictVars = {}
        Variables(self, model)

    def solveMINLP(self, solver, pyoDict):

        # initialize class to store algorithm performance metrics
        performanceInstance = Performance(self)
        # initialize class to print performance metrics
        outputMaster = Output(self, performanceInstance)
        for run in self.nlpDict['hyperparameters']['runsNumberArray']:
            # initialize the class containing all the methods for the generation and modification of solutions
            solutionsInstance = Solutions(self, run)
            for iteration in self.nlpDict['hyperparameters']['iterationsNumberArray']:
                if iteration == 0:
                    step = ''
                    # create the solution archive with random assignment
                    solutionsIndices, SA = solutionsInstance.solutionSets(step)
                else:
                    step = 'new'
                    # modify the solution archive according to pdf of solutions
                    solutionsIndices, SA = solutionsInstance.solutionSets(step)

                for solutionIndex in solutionsIndices:
                    # input variables to the MILP model
                    valuesContinuousVariables, valuesDicreteVariables = SA['R'][solutionIndex,:], SA['O'][solutionIndex, :]
                    # update the Pyomo dictionary with the values of the nonlinear variables at current iteration
                    pyoDict = solutionsInstance.updateMILPDict(pyoDict, valuesContinuousVariables, valuesDicreteVariables)
                    # solve the slave problem based on the values of the nonlinear variables at current iteration
                    # self.model.solve(solver, pyoDict)
                    # update the objective function based on the results of the slave MILP problem
                    # solutionsInstance.updateObjective(self.model.instance, solutionIndex, step)

                # rank the solutions according to the computed objective function and select the best among them
                solutionsInstance.rank(step)

                # record the solution
                performanceInstance.record(solutionsInstance)
                if self.nlpDict['hyperparameters']['convergence']['check']:
                    performanceInstance.checkConvergence(iteration)

                # check convergence and print variables to file
                if performanceInstance.converged:
                    outputMaster.reportConvergence(run, iteration, solutionsInstance)
                    break

            if (self.nlpDict['hyperparameters']['convergence']['restart'] and
                    (iteration != self.nlpDict['hyperparameters']['iterationsNumberArray'][-1])):
                # re-initialize the solution archive with memory of the optimum found
                performanceInstance.restart(iteration, solutionsInstance)
                #TODO: add the routines follwing restart

            elif iteration != self.nlpDict['hyperparameters']['iterationsNumberArray'][-1]:
                outputMaster.maxFunctionEvaluationsAchieved()

            # print to file data current run
            outputMaster.fileRun(run)
            # initialize the performance metrics
            performanceInstance.newRun()

        # print to file data current run
        outputMaster.fileRuns()
        outputMaster.reportRuns()