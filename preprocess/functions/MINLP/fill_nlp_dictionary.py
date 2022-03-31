"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class to add to pyomo dictionary the nonlinear functions
==========================================================================================================================================================================="""
import numpy as np
from preprocess.functions.add_parameters     import add_function
from preprocess.functions.extract_input_data import DataInput
from scipy.interpolate                       import interp1d

class FillNlpDict:
    
    def functionNonlinearApproximation(self):
        self.nlpDict['data'] = {}
        technologySubsets    = ['setConversionTechnologies']

        for technologySubset in technologySubsets:
            for technologyName in self.system[technologySubset]:
                for parameterName in self.data[technologySubset][technologyName]:
                        if 'nonlinear' in parameterName:
                            if technologyName in self.analysis["nonlinearTechnologyApproximation"][parameterName.replace("nonlinear","")]:
                                x = self.data[technologySubset][technologyName][parameterName]['capacity'].values
                                y = self.data[technologySubset][technologyName][parameterName]['capex'].values
                                                    
                                # key to use in the Pyomo dictionary
                                key = (technologyName)

                                # add the function to the Pyomo dictionary based on the key and the function object
                                add_function(self.nlpDict['data'], interp1d(x, y, kind='linear'), key, parameterName)


    def configSolver(self):
        self.nlpDict['hyperparameters'] = {}

        # derive parameters from those in config.solver
        FEMax = self.solver['parametersMetaheuristic']['FEsMax']
        k     = self.solver['parametersMetaheuristic']['kNumber']
        m     = self.solver['parametersMetaheuristic']['mNumber']
        self.solver['parametersMetaheuristic']['iterationsNumber'] = np.int(FEMax/(k+m))

        # add parameters from solver or create arrays based on them
        for parameterName in self.solver['parametersMetaheuristic'].keys():
            if 'Number' in parameterName:
                # create array of integers with length given by the input parameter
                object = np.arange(self.solver['parametersMetaheuristic'][parameterName], dtype=np.int)
                key = parameterName+'Array'
            else:
                object = self.solver['parametersMetaheuristic'][parameterName]
                key = parameterName

            # add the element to the dictionary based on the respective key
            self.nlpDict['hyperparameters'][key] = object

    def collectDomainExtremes(self):

        # create DataInput object
        self.dataInput = DataInput(self.system,self.analysis,self.solver)
        
        self.nlpDict['data']['LB'] = {}
        self.nlpDict['data']['UB'] = {}
        self.nlpDict['data']['DS'] = {}

        for variableName in self.analysis['variablesNonlinearModel']:
            for technologyName in self.analysis['variablesNonlinearModel'][variableName]:
                if variableName == 'builtCapacity':
                    for setName in ['setConversionTechnologies', 'setStorageTechnologies', 'setTransportTechnologies']:
                        if technologyName in self.system[setName]:
                            _inputPath = self.paths[setName][technologyName]["folder"]
                            self.nlpDict['data']['LB'][variableName + technologyName] = self.dataInput.extractAttributeData(_inputPath,"minBuiltCapacity")
                            self.nlpDict['data']['UB'][variableName + technologyName] = self.dataInput.extractAttributeData(_inputPath,"maxBuiltCapacity")
                            # self.nlpDict['data']['DS'][variableName + technologyName] = self.dataInput.extractAttributeData(_inputPath,"deltaBuiltCapacity")

    def add_parameter(dictionary, df, dfIndexNames, dfIndex, dfColumn, key, parameter, element=None):

        if df.empty:
            pass

        else:
            df = df.set_index(dfIndexNames)

            value = df.loc[dfIndex, dfColumn]

            if element:
                # if no element specified
                if parameter not in dictionary:
                    # create a new dictionary for the parameter
                    dictionary[parameter] = {element: {key: value}}
                else:
                    if element not in dictionary[parameter]:
                        # create a new dictionary for the element in the parameter
                        dictionary[parameter][element] = {key: value}
                    else:
                        # add the indexes to the dictionary
                        dictionary[parameter][element][key] = value
            else:
                if parameter not in dictionary:
                    # create a new dictionary for the parameter
                    dictionary[parameter] = {key: value}
                else:
                    # add the indexes to the dictionary
                    dictionary[parameter][key] = value

    def add_function(dictionary, function, key, parameterName):

        if parameterName not in dictionary:
            # create a new dictionary for the function
            dictionary[parameterName] = {key: function}
        else:
            # add the indexes to the dictionary
            dictionary[parameterName][key] = function
