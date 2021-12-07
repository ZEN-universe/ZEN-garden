"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class to modify the config dictionary based on existing inputs from config and default_config.
==========================================================================================================================================================================="""

#%% CLASS DEFINITION AND METHODS
class UpdateConfig:
    
    def createSetsFromSubsets(self):
        """
        This method creates new list of sets from subsets' keys
        """
        # create a new list per set name
        for setName in self.analysis['subsets'].keys():
            self.system[setName] = []
        
        # extend the list of elements in the set with all the items of the single subset
        for setName in self.analysis['subsets'].keys():
            for subsetName in self.analysis['subsets'][setName]:
                self.system[setName].extend(self.system[subsetName])
                

    def createSupportPoints(self):
        """
        This method creates sets of support points for PWA
        """
        technologySubset = 'setProductionTechnologies'
        parameterNames   = ['CapexPWA']
        
        # add a set containing the supporting points of the cost
        for technologyName in self.system[technologySubset]:
            for parameterName in parameterNames:
                df = self.data[technologySubset][technologyName][parameterName]
                
                # create a new set with the indexes of the supporting points
                setName = 'set'+parameterName
                self.system[setName] = list(df[self.analysis['dataInputs']['PWA']['supportPoints']].values)
                