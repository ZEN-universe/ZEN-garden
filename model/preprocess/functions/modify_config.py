"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class to modify the config dictionary based on existing inputs from config and default_config
==========================================================================================================================================================================="""

class UpdateConfig:
    
    def __init__(self):
        pass
    
    def createSetsFromSubsets(self):
        
        # create a new list per set name
        for setName in self.analysis['subsets'].keys():
            self.system[setName] = []
        
        # extend the list of elements in the set with all the items of the single subset
        for setName in self.analysis['subsets'].keys():
            for subsetName in self.analysis['subsets'][setName]:
                self.system[setName].extend(self.system[subsetName])