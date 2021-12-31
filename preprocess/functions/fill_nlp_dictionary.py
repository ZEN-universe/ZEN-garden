"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class to add to pyomo dictionary the nonlinear functions
==========================================================================================================================================================================="""
from preprocess.functions.add_parameters import add_function
from scipy.interpolate import interp1d

class FillNlpDict:
    
    def __init__(self):
        pass                  
    
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
                        add_function(self.nlpDict[None], interp1d(x, y, kind='linear'), key, parameterName)