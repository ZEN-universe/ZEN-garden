from model.objects.element import Element
import itertools
import numpy as np

class Variables:

    def __init__(self, object, model, nlpDict):

        # inherit attributes from parent class
        self.model = model.model
        self.analysis = object.analysis
        self.system = object.system
        self.dictVars = object.dictVars
        self.nlpDict = nlpDict

        # define the variables input and output of the master algorithm
        self.collectVariables()
        # collect the attributes necessary to handle the solution archive
        self.createInputsArchive()

    def collectVariables(self):
        """"create a dictionary containing the set of variables subject to nonlinearities.
        :return: dictionary containing two main dictionaries. ('variablesInput') dictionary with domain and name of
            variables input to the metaheuristic algorithm. ('variablesOutput') dictionary with the variables output
            of the metaheuristic algorithm and input to the MILP problem.
        """

        if [self.analysis['nonlinearTechnologyApproximation'][type] for type
            in self.analysis['nonlinearTechnologyApproximation'].keys()]:
            self.dictVars['output'] = {}
            for type in self.analysis['nonlinearTechnologyApproximation'].keys():
                if type == 'Capex':
                    variableName = 'capex'
                elif type == 'ConverEfficiency':
                    variableName = 'converEfficiency'

                # if there are technologies in the list extract the dimensions and the domain from the declaration
                # in the model instance
                if self.analysis['nonlinearTechnologyApproximation'][type]:
                    for technologyName in self.analysis['nonlinearTechnologyApproximation'][type]:
                        name = variableName + technologyName
                        _, dimensions, domain = Element(self).getProperties(getattr(self.model, name).doc)
                        self.storeAttributes(self.dictVars['output'], name, dimensions, domain)
        else:
            raise ValueError('No output variables expected from master algorithm')

        # add the variables input to the master algorithm
        if self.analysis['variablesNonlinearModel']:
            self.dictVars['input'] = {}
            for variableName in self.analysis['variablesNonlinearModel']:
                for technologyName in self.analysis['variablesNonlinearModel'][variableName]:
                    name = variableName + technologyName
                    _, dimensions, domain = Element(self).getProperties(getattr(self.model, name).doc)
                    self.storeAttributes(self.dictVars['input'], name, dimensions, domain)

        else:
            raise ValueError('No input variables to master algorithm')

    @staticmethod
    def storeAttributes(dictionary, name, dimensions, domain):
        """"static method to add elements to dictionary
        :return: dictionary with additional keys
        """
        # extract the typology of domain
        dictionary[name] = {
            'dimensions': [dim.local_name for dim in dimensions],
            'domain': domain.local_name
        }

    def createInputsArchive(self):
        for varType in ['R', 'O']:
            self.dictVars[varType] = {'names': []}

            # split the input variables based on their domain
        for variableName in self.dictVars['input'].keys():
            domain = self.dictVars['input'][variableName]['domain']
            # collect all the indices
            variableIndices = []
            for setName in self.dictVars['input'][variableName]['dimensions']:
                variableIndices.extend([self.system[setName]])
            # create a new variable name per combination of indices
            for indexTuple in itertools.product(*variableIndices):
                name = variableName + '_' + '_'.join(map(str, indexTuple))
                if 'Real' in domain:
                    self.dictVars['R']['names'].append(name)
                elif ('Integer' in domain) or ('Boolean' in domain):
                    self.dictVars['O']['names'].append(name)

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
                    self.dictVars[type]['LBArray'][0, idx] = self.nlpDict['LB'][name]
                    self.dictVars[type]['UBArray'][0, idx] = self.nlpDict['UB'][name]
                elif type == 'O':
                    self.dictVars[type]['LBArray'][0, idx] = 0
                    self.dictVars[type]['UBArray'][0, idx] = np.int(self.nlpDict['UB'][name]/self.nlpDict['DS'][name])