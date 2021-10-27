# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                               Labratory of Risk and Reliability Engineering
#                                         ETH Zurich, September 2021

# ======================================================================================================================

import os           
    
def Carriers(self):
    
    self.input['carriers'] = dict()
    path = self.paths['carriers']['folder']
    
    # read all the folders in the carriers directory
    for carrier in next(os.walk(path))[1]:
        self.input['carriers'][carrier] = dict()

        
def Networks(self):

    self.input['networks'] = dict() 
    path = self.paths['networks']['folder']
    
    # read all the folders in the networks directory
    for network in next(os.walk(path))[1]:
        self.input['networks'][network] = dict()
        
def Technologies(self):
    
    technology_types = ['production_technologies',\
                        'storage_technologies',\
                        'transport_technologies'
                        ]
    
    for technology_type in technology_types:
        
        self.input[technology_type] = dict()
        path = self.paths[technology_type]['folder']
        
        # read all the folders in the directory of the specific type of 
        # technology
        for technology in next(os.walk(path))[1]:
            self.input[technology_type][technology] = dict()         
        
                