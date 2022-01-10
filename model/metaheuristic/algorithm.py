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
import itertools
import numpy as np

class Metaheuristic(Model, Element):

    def __init__(self, model, config, nlpDict):

        logging.info('initialize metaheuristic algorithm inheriting the methods from class Model and Element')

        # instantiate model
        self.model = model.model
        # instantiate analysis
        self.analysis = config.analysis
        # instantiate system
        self.system = config.system
        # instantiate solver
        self.solver = config.solver
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
        """"create a dictionary containing the set of variables subject to nonlinearities.
        :return: dictionary containing two main dictionaries. ('variablesInput') dictionary with domain and name of
            variables input to the metaheuristic algorithm. ('variablesOutput') dictionary with the variables output
            of the metaheuristic algorithm and input to the MILP problem.
        """

        if [self.analysis['nonlinearTechnologyApproximation'][type] for type
            in self.analysis['nonlinearTechnologyApproximation'].keys()]:
            self.dictVars = {'output': {}}
            for type in self.analysis['nonlinearTechnologyApproximation'].keys():
                if type == 'Capex':
                    variableName = 'capex'
                elif type == 'ConverEfficiency':
                    variableName = 'converEfficiency'

                # if there are technologies in the list extract the dimensions and the domain from the declaration
                # in the model instance
                if self.analysis['nonlinearTechnologyApproximation'][type]:
                    for technologyName in self.analysis['nonlinearTechnologyApproximation'][type]:
                        self.storeAttributes(self.dictVars['output'], variableName+technologyName)
        else:
            raise ValueError('No output variables expected from master algorithm')

        # add the variables input to the master algorithm
        if self.analysis['variablesNonlinearModel']:
            self.dictVars['input']= {}
            for variableName in self.analysis['variablesNonlinearModel']:
                for technologyName in self.analysis['variablesNonlinearModel'][variableName]:
                    self.storeAttributes(self.dictVars['input'], variableName + technologyName)

            for varType in ['R', 'O']:
                self.dictVars[varType] = {'names':[]}

            # split the input variables based on their domain
            for variableName in self.dictVars['input'].keys():
                domain = self.dictVars['input'][variableName]['domain']
                self.dictVars['input'][variableName]['elements'] = []
                # collect all the indices
                variableIndices = []
                for setName in self.dictVars['input'][variableName]['dimensions']:
                        variableIndices.extend([self.system[setName]])
                # create a new variable name per combination of indices
                for indexTuple in itertools.product(*variableIndices):
                    name = variableName+'_'+'_'.join(map(str, indexTuple))
                    if 'Real' in domain:
                        self.dictVars['R']['names'].append(name)
                    elif ('Integer' in domain) or ('Boolean' in domain):
                        self.dictVars['O']['names'].append(name)

            # derive parameters related to the variable order, size and indices
            for type in ['R','O']:
                self.dictVars[type]['n'] = len(self.dictVars[type]['names'])
                self.dictVars[type]['idx'] = np.arange(self.dictVars[type]['n'],  dtype=np.int)
                self.dictVars[type]['name_idx'] = dict(zip(self.dictVars[type]['names'], self.dictVars[type]['idx']))
                self.dictVars[type]['idx_name'] = dict(zip(self.dictVars[type]['idx'], self.dictVars[type]['names']))
        else:
            raise ValueError('No input variables to master algorithm')

    def storeAttributes(self, dictionary, name):
        # extract the typology of domain
        varObject = getattr(self.model, name)
        print(varObject.doc)
        _, dimensions, domain = self.getProperties(varObject.doc)
        dictionary[name] = {
            'dimensions': [dim.local_name for dim in dimensions],
            'domain': domain.local_name
        }