# =====================================================================================================================
"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class to initialise the dictionary to store the input data.
==========================================================================================================================================================================="""

# IMPORT AND SETUP
import os


#%% CLASS DEFINITION AND METHODS
class Init:
        
    def carriers(self):
        """
        This method initialises the keys with the names of the input carriers
        """
        for carrierSubset in self.analysis['subsets']['setCarriers']:
            self.data[carrierSubset] = dict()
            path = self.paths[carrierSubset]['folder']
            
            # read all the folders in the carriers directory
            for carrierName in next(os.walk(path))[1]:
                self.data[carrierSubset][carrierName] = dict()


    def technologies(self):
        """
        This method initialises the keys with the names of the technologies
        """
        for technologySubset in self.analysis['subsets']['setTechnologies']:         
            self.data[technologySubset] = dict()
            path = self.paths[technologySubset]['folder']
            
            # read all the folders in the directory of the specific type of technology
            for technologyName in next(os.walk(path))[1]:
                self.data[technologySubset][technologyName] = dict()         


    def nodes(self):
        """
        This method initialises the key with the set of nodes
        """
        self.data['setNodes'] = dict()
    

    def times(self):
        """
        This method initialises the key with the set of time steps
        """
        self.data['setTimeSteps'] = dict()   
        

    def scenarios(self):
        """
        This method initialises the key with the set of scenarios
        """
        self.data['setScenarios'] = dict()     
    