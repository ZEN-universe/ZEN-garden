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

class Metaheuristic:

    def __init__(self, model, nlpDict):

        logging.info('initialize metaheuristic algorithm class')

        # instantiate analysis
        self.analysis = model.analysis
        # instantiate system
        self.system = model.system
        # instantiate dictionary containing all the required hyperparamters and input data
        self.nlpDict = nlpDict

        # collect the properties of the decision variables handled by the metaheuristic
        self.dictVars = {}
        Vars = Variables(self, model)

        for run in nlpDict['hyperparameters']['runsNumberArray']:
            # instantiate the class containing all the methods for the generation and modification of solutions
            Solution = Solutions(self, run)

        # for run in nlpDict['runsNumberArray']:
        #     # initialize run
        #     solverInstance.initialize_run(run)
        #     for iteration in nlpDict['iterationsNumberArray']:
        #         # modify the set of solutions
        #         solverInstance.modify(run, iteration)
        #         # update the dictionary with the values of the nonlinear variables at current iteration
        #         pyoDict = solverInstance.updateMILPDict(pyoDict)
        #         # solve the slave problem based on the values of the nonlinear variables at current iteration
        #         # self.solveMILP(solver, pyoDict)
        #         # update the objective function based on the results of the slave MILP problem
        #         # solverInstance.updateSolution(self.model.instance)

