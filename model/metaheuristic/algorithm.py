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
from model.model import Model
from model.objects.element import Element

class Metaheuristic(Model, Element):

    def __init__(self, model, analysis, solver, nlpDict):

        logging.info('initialize metaheuristic algorithm inheriting the methods from class Model and Element')

        # instantiate model
        self.model = model.model
        # instantiate analysis
        self.analysis = analysis
        # instantiate solver
        self.solver = solver
        # instantiate dictionary
        self.nlpDict = nlpDict

        # define attributes of the variables handled by the algorithm
        self.collectVariables()

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

    def collectVariables(self):
        self.dictVars = {}
        for type in self.analysis['nonlinearTechnologyApproximation'].keys():
            if type == 'Capex':
                variableName = 'capex'

            # if there are technologies in the list extract the dimensions and the domain from the declaration
            # in the model instance
            if self.analysis['nonlinearTechnologyApproximation'][type]:
                for technologyName in self.analysis['nonlinearTechnologyApproximation'][type]:

                    varObject = getattr(self.model, variableName+technologyName)
                    print(varObject.doc)
                    _, dimensions, domain = self.getProperties(varObject.doc)
                    self.dictVars[variableName+technologyName] = {
                        'dimensions': [dim.local_name for dim in dimensions],
                        'domain': domain.local_name
                    }
            #TODO: introduce variable domain in dataset per technology e.g. attributes (add discretizaiton step for ordinate variables)