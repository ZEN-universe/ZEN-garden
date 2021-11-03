# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                               Labratory of Risk and Reliability Engineering
#                                         ETH Zurich, September 2021

# ======================================================================================================================

import os

class Init:
    
    def __init__(self):
        pass
        
    def carriers(self):
        
        for carrierSubset in self.system['carrierSubsets']:
            self.data[carrierSubset] = dict()
            path = self.paths[carrierSubset]['folder']
            
            # read all the folders in the carriers directory
            for carrierName in next(os.walk(path))[1]:
                self.data[carrierSubset][carrierName] = dict()
            
    def technologies(self):
    
        for technologySubset in self.system['technologySubsets']:
            
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
    
    