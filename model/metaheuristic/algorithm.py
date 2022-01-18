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

        # collect the properties of the decision variables handled by the metaheuristic and create a new attribute in
        # the Pyomo dictionary
        self.dictVars = {}
        Variables(self, model)

        # initialize class to store algorithm performance metrics
        performanceInstance = Performance(self)
        # initialize class to print performance metrics
        outputMaster = Output(self, performanceInstance)
        for run in nlpDict['hyperparameters']['runsNumberArray']:
            # initialize the class containing all the methods for the generation and modification of solutions
            solutionsInstance = Solutions(self, run)
            for iteration in nlpDict['iterationsNumberArray']:
                if iteration == 0:
                    step = ''
                    # create the solution archive with random assignment
                    solutionsIndices, SA_r, SA_o = solutionsInstance.solutionSets(step)
                else:
                    step = 'new'
                    # modify the solution archive according to pdf of solutions
                    solutionsIndices, SA_r, SA_o = solutionsInstance.solutionSets(step)

                for solutionIndex in solutionsIndices:
                    # input variables to the MILP model
                    valuesContinuousVariables = SA_r[solutionIndex,:]
                    valuesDicreteVariables = SA_o[solutionIndex, :]
                    # update the Pyomo dictionary with the values of the nonlinear variables at current iteration
                    pyoDict = solutionsInstance.updateMILPDict(pyoDict, valuesContinuousVariables, valuesDicreteVariables)

                    # solve the slave problem based on the values of the nonlinear variables at current iteration for
                    # values in the solution archive at row solutionIndex
                    model.solve(model.solver, pyoDict)
                    # update the objective function based on the results of the slave MILP problem
                    solutionsInstance.updateObjective(model.instance, solutionIndex, step)
                # rank the solutions according to the computed objective function and select the best among them
                solutionsInstance.rank(step)

                # record the solution
                performanceInstance.record(solutionsInstance)
                if self.nlpDict['parametersMetaheuristic']['convergence']['check']:
                    performanceInstance.checkConvergence(iteration)

                # check convergence and print variables to file
                if performanceInstance.converged:
                    outputMaster.reportConvergence(run, iteration)
                    break
