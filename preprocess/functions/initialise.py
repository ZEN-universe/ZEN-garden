# =====================================================================================================================
"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class to initialise the dictionary to store the input data.
==========================================================================================================================================================================="""

import os

class Init:
    
    def __init__(self):
        pass
        
    def carriers(self):
        
        for carrierSubset in self.analysis['subsets']['setCarriers']:
            self.data[carrierSubset] = dict()
            path = self.paths[carrierSubset]['folder']
            
            # read all the folders in the carriers directory
            for carrierName in next(os.walk(path))[1]:
                self.data[carrierSubset][carrierName] = dict()
            
    def technologies(self):
    
        for technologySubset in self.analysis['subsets']['setTechnologies']:
            
            self.data[technologySubset] = dict()
            path = self.paths[technologySubset]['folder']
            
            # read all the folders in the directory of the specific type of 
            # technology
            for technologyName in next(os.walk(path))[1]:
                self.data[technologySubset][technologyName] = dict()         
            
    def nodes(self):
        
        self.data['setNodes'] = dict()
    
    def times(self):
        
        self.data['setTimeSteps'] = dict()   
        
    def scenarios(self):
        
        self.data['setScenarios'] = dict()     
    
    