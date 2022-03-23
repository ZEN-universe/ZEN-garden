"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      January-2022
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class interfacing the declaration of variables and parameters in the slave algorithm with the master algorithm.
==========================================================================================================================================================================="""

from model.objects.element import Element
import itertools
import numpy as np

class Variables:

    def __init__(self, object, model):

        # inherit attributes from parent class
        self.model = model.model
        self.analysis = object.analysis
        self.system = object.system
        self.dictVars = object.dictVars
        self.nlpDict = object.nlpDict

        # define the variables input and output of the master algorithm as declared in the slave algorithm
        self.collectVariables()
        # collect the attributes necessary to handle the solution archive
        self.createInputsSolutionArchive()

    def collectVariables(self):
        """"create a dictionary containing the set of variables subject to nonlinearities.
        :return: dictionary containing the domain and name of variables handled by the metaheuristic algorithm
        """

        # add the variables input to the master algorithm
        if self.analysis['variablesNonlinearModel']:
            self.dictVars['input'] = {}
            for variableName in self.analysis['variablesNonlinearModel']:
                for technologyName in self.analysis['variablesNonlinearModel'][variableName]:
                    # extract variable name from model object with associated indexes
                    indexedVariable = self.model.find_component(variableName)

                    # loop through all the indices of the variable to create one variable per index
                    for index in indexedVariable:
                        if technologyName in index:
                            variable = indexedVariable[index]
                            self.dictVars['input'][variableName+"_".join(map(str,index))] = {}
                            self.dictVars['input'][variableName+"_".join(map(str,index))]['variable'] = variable
                            self.dictVars['input'][variableName+"_".join(map(str,index))]['domain'] = variable.domain.local_name
        else:
            raise ValueError('No input variables to master algorithm')

    def createInputsSolutionArchive(self):
        """collect inputs in the format of the solution archive concerning the variables domain.
        :return:  dictionary with additional keys
        """
        for varType in ['R', 'O']:
            self.dictVars[varType] = {'names': []}

        # split the input variables based on their domain
        for variableName in self.dictVars['input'].keys():
            domain = self.dictVars['input'][variableName]["domain"]
            if 'Real' in domain:
                self.dictVars['R']['names'].append(variableName)
            elif ('Integer' in domain) or ('Boolean' in domain):
                self.dictVars['O']['names'].append(variableName)
            
        # derive parameters related to the variable order, size, indices, lower and upper bounds
        for type in ['R', 'O']:
            self.dictVars[type]['n'] = len(self.dictVars[type]['names'])
            self.dictVars[type]['idxArray'] = np.arange(self.dictVars[type]['n'], dtype=np.int)
            self.dictVars[type]['name_to_idx'] = dict(zip(self.dictVars[type]['names'], self.dictVars[type]['idxArray']))
            self.dictVars[type]['idx_to_name'] = dict(zip(self.dictVars[type]['idxArray'], self.dictVars[type]['names']))
            self.dictVars[type]['LBArray'] = np.zeros([1, self.dictVars[type]['n']])
            self.dictVars[type]['UBArray'] = np.zeros([1, self.dictVars[type]['n']])
            for idx in self.dictVars[type]['idxArray']:
                name = self.dictVars[type]['idx_to_name'][idx].split('_')[0]
                if type == 'R':
                    self.dictVars[type]['LBArray'][0, idx] = self.nlpDict['data']['LB'][name]
                    self.dictVars[type]['UBArray'][0, idx] = self.nlpDict['data']['UB'][name]
                elif type == 'O':
                    self.dictVars[type]['LBArray'][0, idx] = 0
                    self.dictVars[type]['UBArray'][0, idx] = np.int(self.nlpDict['data']['UB'][name]/self.nlpDict['data']['DS'][name])
                    self.dictVars[type]['values'][0, idx] = \
                        np.arange(self.dictVars[type]['UBArray'][0, idx], dtype=np.int)*self.nlpDict['data']['DS'][name]