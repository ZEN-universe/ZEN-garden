"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      January-2022
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class containing the metaheuristic algorithm.
              The algorithm takes as input the set of decision variables subject to nonlinearities and the respective domains.
              Iteratively the predicted values of the decision variables are modified and only those associated to the highest quality solutions are selected.
              The values of the decision variables are then passed to the dictionary of the MILP solver as constant values in the model.
==========================================================================================================================================================================="""
import logging
import numpy as np
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

        # collect the properties of the decision variables handled by the metaheuristic
        self.dictVars = {}
        Variables(self, model)

    def solveMINLP(self, solver):

        # instantiate the solver
        self.solver = solver
        # initialize class to store algorithm performance metrics
        performanceInstance = Performance(self)
        # initialize class to print performance metrics
        outputMaster = Output(self, performanceInstance)
        for run in self.nlpDict['hyperparameters']['runsNumberArray']:
            if run % self.solver['performanceCheck']['printDeltaRun']  == 0:
                print(f'run {run}')
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
                    valuesContinuousVariables, valuesDiscreteVariables = SA['R'][solutionIndex,:], SA['O'][solutionIndex, :]
                    # update slave concrete model and solve it
                    self.slaveAlgorithm(solutionsInstance, valuesContinuousVariables, valuesDiscreteVariables, solutionIndex, step)

                # rank the solutions according to the computed objective function and select the best among them
                solutionsInstance.rank(step)

                if iteration % self.solver['performanceCheck']['printDeltaIteration'] == 0:
                    print(f'iteration {iteration}')
                # record the solution
                performanceInstance.record(solutionsInstance)
                if self.solver['convergenceCriterion']['check']:
                    performanceInstance.checkConvergence(iteration)

                # check convergence and print variables to file
                if performanceInstance.converged:
                    outputMaster.reportConvergence(run, iteration, solutionsInstance)
                    break

            if (self.solver['convergenceCriterion']['restart'] and
                    (iteration != self.nlpDict['hyperparameters']['iterationsNumberArray'][-1])):
                # re-initialize the solution archive with memory of the optimum found
                solutionsIndices, SA = performanceInstance.restart(iteration, solutionsInstance)

                for solutionIndex in solutionsIndices:
                    # input variables to the MILP model
                    valuesContinuousVariables, valuesDiscreteVariables = SA['R'][solutionIndex,:], SA['O'][solutionIndex, :]
                    # update slave concrete model and solve it
                    self.slaveAlgorithm(solutionsInstance, valuesContinuousVariables, valuesDiscreteVariables, solutionIndex, step='')
                # rank the solutions according to the computed objective function and select the best among them
                solutionsInstance.rank(step='')

                for iteration in self.nlpDict['hyperparameters']['iterationsNumberArray'][performanceInstance.iteration0:]:
                    # modify the solution archive according to pdf of solutions
                    solutionsIndices, SA = solutionsInstance.solutionSets(step='new')
                    for solutionIndex in solutionsIndices:
                        # input variables to the MILP model
                        valuesContinuousVariables, valuesDiscreteVariables = SA['R'][solutionIndex, :], SA['O'][solutionIndex,:]
                        # update slave concrete model and solve it
                        self.slaveAlgorithm(solutionsInstance, valuesContinuousVariables, valuesDiscreteVariables,
                                            solutionIndex, step='new')

                    solutionsInstance.rank(step='new')
                    if iteration % self.solver['performanceCheck']['printDeltaIteration'] == 0:
                        print('iteration', iteration)
                    # record the solution
                    performanceInstance.record(solutionsInstance)
                    if self.solver['convergenceCriterion']['check']:
                        performanceInstance.checkConvergence(iteration)

                    # check convergence and print variables to file
                    if performanceInstance.converged:
                        outputMaster.reportConvergence(run, iteration, solutionsInstance)
                        break

            elif iteration == self.nlpDict['hyperparameters']['iterationsNumberArray'][-1]:
                outputMaster.maxFunctionEvaluationsAchieved()

            # print to file data current run
            outputMaster.fileRun(run)
            # store solution current run and initialize the performance metrics
            performanceInstance.newRun()

            #TODO: add storage of solution instance from run

        # print to file data current run
        outputMaster.fileRuns()
        outputMaster.reportRuns()

    def slaveAlgorithm(self, solutionsInstance, valuesContinuousVariables, valuesDiscreteVariables, solutionIndex, step):
        """ handles the interface with the slave algorithm
        :param solutionsInstance: instance object containing elements of current solution
        :param valuesContinuousVariables: obtained values for the continuous variables
        :param valuesDiscreteVariables: obtained values for the discrete variables
        :param solutionIndex: index of current solution in solution archive and objective function array
        :param step: flag identifying the solution archive
        """
        # update the Pyomo dictionary with the values of the nonlinear variables at current iteration
        self.fixMILPVariables(valuesContinuousVariables, valuesDiscreteVariables)
        # solve the slave problem based on the values of the nonlinear variables at current iteration
        self.model.solve(self.solver)
        # update the objective function based on the results of the slave MILP problem
        solutionsInstance.updateObjective(self.model.model, solutionIndex, step)

    def fixMILPVariables(self,valuesContinuousVariables,valuesDiscreteVariables):
        """ fixes the variables calculated by the meta-heuristics to the obtained value in the MILP 
        :param valuesContinuousVariables: obtained values for the continuous variables
        :param valuesDiscreteVariables: obtained values for the discrete variables"""
        
        # continuous variables
        for idxContinuousVariable in self.dictVars["R"]["idx_to_name"]:
            continuousVariable = self.dictVars["R"]["idx_to_name"][idxContinuousVariable]
            valueContinuousVariable = valuesContinuousVariables[idxContinuousVariable]
            variableInModel = self.dictVars["input"][continuousVariable]["variable"]
            # fix variable value
            variableInModel.fix(valueContinuousVariable)
            # fix Capex as well
            tech = variableInModel.index()[0]
            if tech in self.nlpDict["data"]["nonlinearCapex"] and tech in self.model.model.setNLCapexTechs:
                interpObject = self.nlpDict["data"]["nonlinearCapex"][tech]
                capexVariableInModel = self.model.model.find_component("capex")[variableInModel.index()]
                capexVariableInModel.fix(interpObject(valueContinuousVariable).item()*valueContinuousVariable)

        # discrete variables
        for idxDiscreteVariable in self.dictVars["O"]["idx_to_name"]:
            discreteVariable = self.dictVars["O"]["idx_to_name"][idxDiscreteVariable]
            valueDiscreteVariable = valuesDiscreteVariables[idxDiscreteVariable]
            variableInModel = self.dictVars["input"][discreteVariable]
            # fix variable value
            variableInModel.fix(valueDiscreteVariable)


