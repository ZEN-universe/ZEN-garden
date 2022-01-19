"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class to add to pyomo dictionary the nonlinear functions
==========================================================================================================================================================================="""
from preprocess.functions.add_parameters import add_function
from preprocess.functions.extract_input_data import DataInput
from scipy.interpolate import interp1d
import numpy as np

class FillNlpDict:
    
    def functionNonlinearApproximation(self):
            
        technologySubsets = ['setConversionTechnologies']

        for technologySubset in technologySubsets:
            for technologyName in self.system[technologySubset]:
                for parameterName in self.data[technologySubset][technologyName]:
                    
                    if 'nonlinear' in parameterName:
                        x = self.data[technologySubset][technologyName][parameterName]['x'].values
                        y = self.data[technologySubset][technologyName][parameterName]['y'].values
                                            
                        # key to use in the Pyomo dictionary
                        key = (technologyName)
                        # add the function to the Pyomo dictionary based on the key and the function object
                        add_function(self.nlpDict, interp1d(x, y, kind='linear'), key, parameterName)


    def configSolver(self):

        # derive parameters from those in config.solver
        FEMax = self.solver['parametersMetaheuristic']['FEsMax']
        k = self.solver['parametersMetaheuristic']['kNumber']
        m = self.solver['parametersMetaheuristic']['mNumber']
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
            self.nlpDict[key] = object

        if self.analysis['sense'] == 'minimize':
            self.nlpDict['penalty'] = np.inf
        elif self.analysis['sense'] == 'maximize':
            self.nlpDict['penalty'] = -np.inf

    def collectDomainExtremes(self):

        # create DataInput object
        self.dataInput = DataInput(self.system,self.analysis)
        
        self.nlpDict['LB'] = {}
        self.nlpDict['UB'] = {}

        for variableName in self.analysis['variablesNonlinearModel']:
            for technologyName in self.analysis['variablesNonlinearModel'][variableName]:
                if variableName == 'capacity':
                    for setName in ['setConversionTechnologies', 'setStorageTechnologies', 'setTransportTechnologies']:
                        if technologyName in self.system[setName]:
                            _inputPath = self.paths[setName][technologyName]["folder"]
                            self.nlpDict['LB'][variableName+technologyName] = self.dataInput.extractAttributeData(_inputPath,"minCapacity")
                            self.nlpDict['UB'][variableName+technologyName] = self.dataInput.extractAttributeData(_inputPath,"maxCapacity")

